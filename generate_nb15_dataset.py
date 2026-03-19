import pandas as pd
import numpy as np
import torch
import joblib
import os
import hashlib
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

def _subnet_key(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) < 3:
            return (0, 0, 0)
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)

def get_ip_id_hash(ip_str):
    hash_obj = hashlib.md5(str(ip_str).encode())
    return int(hash_obj.hexdigest()[:15], 16)

def get_subnet_id_safe(ip_str, subnet_map):
    key = _subnet_key(ip_str)
    return subnet_map.get(key, 0)

# === 新增：图标签统计函数 ===
def print_graph_label_stats(graph_seq, split_name, class_names):
    counts = np.zeros(len(class_names), dtype=np.int64)
    for g in graph_seq:
        if g is not None and g.edge_labels is not None:
            labels = g.edge_labels.detach().cpu().numpy().astype(np.int64)
            if len(labels) > 0:
                counts += np.bincount(labels, minlength=len(class_names))
    
    stats = [f"{class_names[i]}({i}): {counts[i]}" for i in range(len(class_names)) if counts[i] > 0]
    print(f"[{split_name} Graphs] Edge Label Counts -> " + ", ".join(stats))

def create_graph_data_inductive(time_slice, subnet_map):
    time_slice = time_slice.copy()
    time_slice['Src IP'] = time_slice['Src IP'].astype(str).str.strip()
    time_slice['Dst IP'] = time_slice['Dst IP'].astype(str).str.strip()

    src_ids = time_slice['Src IP'].apply(get_ip_id_hash).values.astype(np.int64)
    dst_ids = time_slice['Dst IP'].apply(get_ip_id_hash).values.astype(np.int64)
    labels = time_slice['Label'].values.astype(int)

    all_nodes_in_slice = np.concatenate([src_ids, dst_ids])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_ids)]
    dst_local = inverse_indices[len(src_ids):]
    
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        out_degrees.scatter_add_(0, edge_index[0], ones)
        in_degrees.scatter_add_(0, edge_index[1], ones)

    # new
    unique_edges = torch.unique(edge_index, dim=1)
    ones_uniq = torch.ones(unique_edges.size(1), dtype=torch.float)
    uniq_in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    uniq_out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    if unique_edges.size(1) > 0:
        uniq_out_degrees.scatter_add_(0, unique_edges[0], ones_uniq)
        uniq_in_degrees.scatter_add_(0, unique_edges[1], ones_uniq)

    src_port = pd.to_numeric(time_slice.get('Src Port', 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)
    # new

    src_port = pd.to_numeric(time_slice.get('Src Port', 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    pkt_col = None
    for cand in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets']:
        if cand in time_slice.columns:
            pkt_col = cand
            break
    
    if pkt_col is None:
        fwd_pkts = torch.zeros(edge_index.size(1), dtype=torch.float)
    else:
        fwd_pkts = torch.tensor(pd.to_numeric(time_slice[pkt_col], errors='coerce').fillna(0).values, dtype=torch.float)
    
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)

    x = torch.stack([
        torch.log1p(in_degrees), 
        torch.log1p(out_degrees), 
        priv_ratio, 
        node_pkt_sum,
        torch.log1p(uniq_in_degrees),   # 注入服务器画像语义
        torch.log1p(uniq_out_degrees)   # 注入扫描器/异常角色语义
    ], dim=-1).float()
    
    subnet_id = None
    if subnet_map is not None:
        subnet_ids_for_node = {get_ip_id_hash(ip): get_subnet_id_safe(ip, subnet_map) for ip in pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()}
        subnet_id = torch.tensor([subnet_ids_for_node.get(int(h), 0) for h in unique_nodes], dtype=torch.long)

    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port', 'time_idx']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr_vals = np.nan_to_num(edge_attr_vals, nan=0.0, posinf=0.0, neginf=0.0)
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    if edge_index.size(1) > 0:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=torch.tensor(labels, dtype=torch.long), n_id=n_id)
        if subnet_id is not None:
            data.subnet_id = subnet_id
        return data
    return None

def main():
    output_dir = "processed_data/unsw_nb15/"
    os.makedirs(output_dir, exist_ok=True)
    print("Loading NB15 Data...")
    
    data = pd.read_csv("data/CIC-NUSW-NB15/CICFlowMeter_out.csv")
    data['Label'] = data['Label'].astype(str).str.strip().replace('', np.nan)
    data.dropna(subset=['Label', 'Timestamp'], inplace=True)
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    class_names = list(label_encoder.classes_)
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

    print("Processing Time...")
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True, errors='coerce')
    data.dropna(subset=['Timestamp'], inplace=True)
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('20s')

    unique_times = data['time_idx'].drop_duplicates().values
    train_idx = int(len(unique_times) * 0.8)
    val_idx = int(len(unique_times) * 0.9)
    split_time_train = unique_times[train_idx]
    split_time_val = unique_times[val_idx]

    train_df = data[data['time_idx'] < split_time_train].copy()
    val_df = data[(data['time_idx'] >= split_time_train) & (data['time_idx'] < split_time_val)].copy()
    test_df = data[data['time_idx'] >= split_time_val].copy()

    print("Normalizing...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port', 'time_idx']]
    
    for df in [train_df, val_df, test_df]:
        df[feat_cols] = df[feat_cols].fillna(0)
    
    for col in feat_cols:
        if train_df[col].max() > 100:
            train_df[col] = np.log1p(train_df[col].abs())
            val_df[col] = np.log1p(val_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print("Saving flattened data...")
    np.savez(os.path.join(output_dir, "flattened_data.npz"),
             X_train=train_df[feat_cols].values, y_train=train_df['Label'].values,
             X_val=val_df[feat_cols].values, y_val=val_df['Label'].values,
             X_test=test_df[feat_cols].values, y_test=test_df['Label'].values,
             feature_names=feat_cols)

    print("Building Subnet Map...")
    subnet_to_idx = {'<UNK>': 0}
    for ip in pd.concat([train_df['Src IP'], train_df['Dst IP']]).astype(str).str.strip().unique():
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)

    print("Building Graphs...")
    train_seq = [create_graph_data_inductive(group, subnet_to_idx) for _, group in tqdm(train_df.groupby('time_idx', sort=True))]
    val_seq = [create_graph_data_inductive(group, subnet_to_idx) for _, group in tqdm(val_df.groupby('time_idx', sort=True))]
    test_seq = [create_graph_data_inductive(group, subnet_to_idx) for _, group in tqdm(test_df.groupby('time_idx', sort=True))]

    train_seq = [g for g in train_seq if g]
    val_seq = [g for g in val_seq if g]
    test_seq = [g for g in test_seq if g]

    # === 统计并打印图标签信息 ===
    print("\n--- Graph Statistics ---")
    print_graph_label_stats(train_seq, "Train", class_names)
    print_graph_label_stats(val_seq, "Val", class_names)
    print_graph_label_stats(test_seq, "Test", class_names)
    print("------------------------\n")

    torch.save(train_seq, os.path.join(output_dir, "train_graphs.pt"))
    torch.save(val_seq, os.path.join(output_dir, "val_graphs.pt"))
    torch.save(test_seq, os.path.join(output_dir, "test_graphs.pt"))
    
    print("NB15 Data Generation Complete!")

if __name__ == "__main__":
    main()