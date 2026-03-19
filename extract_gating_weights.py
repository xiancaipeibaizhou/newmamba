import os
import re
import argparse
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

# 导入现有的超参和模型组件
from hparams_a3 import resolve_hparams
from model import MILAN
from ablation_models import (
    MILAN_WoGlobal, MILAN_WoLocal, MILAN_WoGating, 
    MILAN_WoEdgeAug, MILAN_StandardTransformer
)

# ==========================================
# 1. 数据加载与拼接逻辑
# ==========================================
class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=10):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.graph_data_seq) - self.seq_len + 1)
    def __getitem__(self, idx):
        return self.graph_data_seq[idx : idx + self.seq_len]

def temporal_collate_fn(batch):
    if len(batch) == 0: return []
    seq_len = len(batch[0])
    batched_seq = []
    for t in range(seq_len):
        graphs_at_t = [sample[t] for sample in batch]
        batched_seq.append(Batch.from_data_list(graphs_at_t))
    return batched_seq

# ==========================================
# 2. 图表绘制函数 (已升级为 Mamba 标签)
# ==========================================
def plot_gating_distribution(df, exp_name, save_dir):
    # 清理类名避免图表乱码
    df['Attack Class'] = df['Attack Class'].apply(lambda x: str(x).replace('\x96', '-'))
    df = df.sort_values(by="Local Stream (Inception)", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    bar_width = 0.6
    x = np.arange(len(df))
    
    # 局部流依然是 Inception
    p1 = plt.bar(x, df['Local Stream (Inception)'], bar_width, color='#FF9999', label='Local Stream Weight ($\u03B1$)')
    # 全局流更新为 Mamba
    p2 = plt.bar(x, df['Global Stream (Mamba)'], bar_width, bottom=df['Local Stream (Inception)'], color='#99CCFF', label='Global Stream Weight ($1-\u03B1$)')
    
    plt.ylabel('Average Gating Weight', fontsize=12, fontweight='bold')
    plt.xlabel('Traffic Type', fontsize=12, fontweight='bold')
    # 更新图表标题
    plt.title('Temporal Module Dependency (Inception vs Mamba) by Attack Type', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, df['Attack Class'], rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right', framealpha=0.9)
    
    for i in range(len(df)):
        local_val = df.iloc[i]['Local Stream (Inception)']
        plt.text(i, local_val / 2, f"{local_val:.2f}", ha='center', va='center', color='black', fontsize=9)
        plt.text(i, local_val + (1 - local_val) / 2, f"{1-local_val:.2f}", ha='center', va='center', color='black', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"gating_distribution_{exp_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 权重分布图已保存至: {save_path}")

# ==========================================
# 3. 核心提取逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract and analyze gating weights from MILAN models.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the best_model.pth file")
    parser.add_argument('--data_dir', type=str, default='../processed_data', help="Directory of processed datasets")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return

    print(f"🔍 解析模型路径: {model_path}")
    
    # 路径解析，提取实验参数
    parts = model_path.split(os.sep)
    dataset_name = parts[-4]
    exp_name = parts[-3]
    model_dir = os.path.dirname(model_path)
    
    match = re.match(r"^(.*?)_(.*)_dim(\d+)_seq(\d+)$", exp_name)
    if not match:
        print(f"❌ 错误: 无法从路径解析实验参数: {exp_name}")
        return
        
    variant, group_str, hidden, seq_len = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
    
    # 加载数据集信息
    dataset_path = os.path.join(args.data_dir, dataset_name)
    print(f"📦 加载测试集: {dataset_name} (seq_len={seq_len})")
    try:
        test_graphs = torch.load(os.path.join(dataset_path, "test_graphs.pt"), weights_only=False)
    except Exception as e:
        print(f"❌ 错误: 无法加载测试集数据: {e}")
        return

    label_enc_path = os.path.join(dataset_path, "label_encoder.pkl")
    if os.path.exists(label_enc_path):
        class_names = joblib.load(label_enc_path).classes_
    else:
        counts = np.zeros(100)
        for g in test_graphs: counts += np.bincount(g.edge_labels.numpy(), minlength=100)
        num_classes = int(np.max(np.nonzero(counts))) + 1
        class_names = [f"Class_{i}" for i in range(num_classes)]
        
    node_dim, edge_dim = test_graphs[0].x.shape[1], test_graphs[0].edge_attr.shape[1]
    test_loader = DataLoader(TemporalGraphDataset(test_graphs, seq_len), batch_size=32, shuffle=False, collate_fn=temporal_collate_fn)

    # 初始化模型
    h = resolve_hparams(group_str, env=os.environ, dataset=dataset_name)
    model_kwargs = {
        "node_in": node_dim, "edge_in": edge_dim, "hidden": hidden, "num_classes": len(class_names),
        "seq_len": seq_len, "heads": int(h["HEADS"]), "dropout": 0.3, "max_cl_edges": int(h.get("MAX_CL_EDGES", 8192)),
        "kernels": list(h["KERNELS"]), "drop_path": float(h.get("DROP_PATH", 0.1)), "dropedge_p": float(h.get("DROPEDGE_P", 0.2)),
    }
    
    if variant == "MILAN": model = MILAN(**model_kwargs).to(device)
    elif variant == "WoGlobal": model = MILAN_WoGlobal(**model_kwargs).to(device)
    elif variant == "WoLocal": model = MILAN_WoLocal(**model_kwargs).to(device)
    elif variant == "WoGating": model = MILAN_WoGating(**model_kwargs).to(device)
    elif variant == "WoEdgeAug": model = MILAN_WoEdgeAug(**model_kwargs).to(device)
    elif variant == "StandardTransformer": model = MILAN_StandardTransformer(**model_kwargs).to(device)
    else: 
        print(f"❌ 错误: 未知的变体 {variant}")
        return

    # 安全加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if not hasattr(model, 'gating'):
        print(f"⚠️ 警告: 当前加载的模型 ({variant}) 不包含 gating 模块，无法提取熵值权重。")
        return

    # 注册前向 Hook 提取 Alpha
    extracted_alphas = []
    def gating_hook(module, input_args, output):
        # 确保拿到的是 tuple 里的第二个返回值 (alpha)
        if isinstance(output, tuple) and len(output) > 1:
            extracted_alphas.append(output[1].detach())
    model.gating.register_forward_hook(gating_hook)

    print("🚀 开始推理并提取时序模块分布权重...")
    all_edge_alphas = []
    all_edge_labels = []

    with torch.no_grad():
        for batched_seq in tqdm(test_loader, desc="Evaluating"):
            extracted_alphas.clear()
            batched_seq_dev = [g.to(device) for g in batched_seq]
            
            _ = model(batched_seq_dev)
            
            # alpha shape: [num_unique_nodes, 1, 1]
            if not extracted_alphas:
                continue
                
            alpha_node = extracted_alphas[0].squeeze(-1).squeeze(-1)
            
            batch_global_ids = []
            for data in batched_seq_dev:
                if hasattr(data, "n_id"): batch_global_ids.append(data.n_id)
                elif hasattr(data, "id"): batch_global_ids.append(data.id)
                else: batch_global_ids.append(torch.arange(data.x.size(0), device=device))
                    
            all_ids = torch.cat(batch_global_ids)
            unique_ids, _ = torch.sort(torch.unique(all_ids))
            
            last_frame = batched_seq_dev[-1]
            indices = torch.searchsorted(unique_ids, batch_global_ids[-1])
            frame_node_alphas = alpha_node[indices]
            
            src, dst = last_frame.edge_index[0], last_frame.edge_index[1]
            edge_alphas = (frame_node_alphas[src] + frame_node_alphas[dst]) / 2.0
            
            all_edge_alphas.append(edge_alphas.cpu().numpy())
            all_edge_labels.append(last_frame.edge_labels.cpu().numpy())

    if not all_edge_alphas:
        print("❌ 提取失败：没有收集到有效的权重数据。")
        return

    # 统计与聚合
    all_edge_alphas = np.concatenate(all_edge_alphas)
    all_edge_labels = np.concatenate(all_edge_labels)
    
    results = []
    for class_idx, class_name in enumerate(class_names):
        mask = (all_edge_labels == class_idx)
        if mask.sum() > 0:
            avg_local_weight = np.mean(all_edge_alphas[mask])
            avg_global_weight = 1.0 - avg_local_weight
            results.append({
                "Attack Class": class_name,
                "Sample Count": mask.sum(),
                "Local Stream (Inception)": avg_local_weight,
                # 列名更新为 Mamba
                "Global Stream (Mamba)": avg_global_weight
            })
            
    df_results = pd.DataFrame(results)
    print("\n✅ 提取完成！各攻击类别下的时序模块平均权重分布：")
    print(df_results.to_string(index=False))

    # 将数据保存为 CSV 并绘制柱状图至模型所在目录
    csv_save_path = os.path.join(model_dir, f"gating_weights_{exp_name}.csv")
    df_results.to_csv(csv_save_path, index=False)
    plot_gating_distribution(df_results, exp_name, model_dir)

if __name__ == "__main__":
    main()

# python extract_gating_weights.py --model_path results/iscx_ids2012/MILAN_DEFAULT_dim64_seq5/20260317-233704/best_model.pth
# MILAN/results/darknet2020_block/MILAN_DEFAULT_dim128_seq10/20260317-235216/best_model.pth
# MILAN/results/cic_ids2017/MILAN_DEFAULT_dim128_seq3/20260317-233003/best_model.pth
# results/darknet2020_block/MILAN_DEFAULT_dim128_seq30/20260318-105715/best_model.pth