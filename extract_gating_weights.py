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
    MILAN_WoEdgeAug, MILAN_StandardTransformer, MILAN_LinearTransformer
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
# 2. 图表绘制函数 
# ==========================================
def plot_gating_distribution(df, exp_name, save_dir):
    df['Attack Class'] = df['Attack Class'].apply(lambda x: str(x).replace('\x96', '-'))
    df = df.sort_values(by="Local Stream (Inception)", ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    bar_width = 0.6
    x = np.arange(len(df))
    
    p1 = ax1.bar(x, df['Local Stream (Inception)'], bar_width, color='#FF9999', label='Local Stream ($\u03B1$)')
    p2 = ax1.bar(x, df['Global Stream (Mamba)'], bar_width, bottom=df['Local Stream (Inception)'], color='#99CCFF', label='Global Stream ($1-\u03B1$)')
    
    ax1.set_ylabel('Average Gating Weight', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Traffic Type', fontsize=12, fontweight='bold')
    ax1.set_title('Temporal Module Dependency & Structural Entropy by Attack Type', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Attack Class'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    
    # ✨ [新增] 在右侧 Y 轴绘制结构熵折线
    ax2 = ax1.twinx()
    p3 = ax2.plot(x, df['Mean Graph Entropy'], color='#2F5597', marker='s', markersize=8, linewidth=2.5, label='Mean Graph Entropy')
    ax2.set_ylabel('Mean Graph Entropy', color='#2F5597', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2F5597')
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', framealpha=0.9)
    
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"gating_distribution_{exp_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 权重与熵值分布图已保存至: {save_path}")

# ==========================================
# 3. 核心提取逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract and analyze gating weights & entropy from MILAN models.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the best_model.pth file")
    parser.add_argument('--data_dir', type=str, default='../processed_data', help="Directory of processed datasets")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return

    print(f"🔍 解析模型路径: {model_path}")
    parts = model_path.split(os.sep)
    dataset_name = parts[-4]
    exp_name = parts[-3]
    model_dir = os.path.dirname(model_path)
    
    match = re.match(r"^(.*?)_(.*)_dim(\d+)_seq(\d+)$", exp_name)
    if not match:
        print(f"❌ 错误: 无法解析实验参数: {exp_name}")
        return
        
    variant, group_str, hidden, seq_len = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
    
    dataset_path = os.path.join(args.data_dir, dataset_name)
    test_graphs = torch.load(os.path.join(dataset_path, "test_graphs.pt"), weights_only=False)

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
    elif variant == "LinearTransformer": model = MILAN_LinearTransformer(**model_kwargs).to(device)
    else: 
        print(f"❌ 错误: 未知的变体 {variant}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if not hasattr(model, 'gating'):
        print(f"⚠️ 警告: 当前加载的模型 ({variant}) 不包含 gating 模块。")
        return

    extracted_alphas = []
    extracted_entropies = []
    
    # ✨ [MR-DID 修改] 注册 Hook 同时提取 Alpha 和 传入的 Entropy 先验
    def gating_hook(module, input_args, output):
        if isinstance(output, tuple) and len(output) > 1:
            extracted_alphas.append(output[1].detach())
            # input_args[3] 是传入的 mean_graph_entropy
            if len(input_args) > 3:
                extracted_entropies.append(input_args[3].detach())
                
    model.gating.register_forward_hook(gating_hook)

    print("🚀 开始推理并提取权重及熵值...")
    all_edge_alphas = []
    all_edge_entropies = []
    all_edge_labels = []

    with torch.no_grad():
        for batched_seq in tqdm(test_loader, desc="Evaluating"):
            extracted_alphas.clear()
            extracted_entropies.clear()
            batched_seq_dev = [g.to(device) for g in batched_seq]
            
            _ = model(batched_seq_dev)
            
            if not extracted_alphas or not extracted_entropies:
                continue
                
            alpha_node = extracted_alphas[0].squeeze(-1).squeeze(-1)
            entropy_val = extracted_entropies[0][0, 0].item() # 获取该批次的标量熵
            
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
            # 为该批次的每一条边复制相同的图熵标量（用于统计归类）
            all_edge_entropies.append(np.full(edge_alphas.shape, entropy_val))
            all_edge_labels.append(last_frame.edge_labels.cpu().numpy())

    all_edge_alphas = np.concatenate(all_edge_alphas)
    all_edge_entropies = np.concatenate(all_edge_entropies)
    all_edge_labels = np.concatenate(all_edge_labels)
    
    results = []
    for class_idx, class_name in enumerate(class_names):
        mask = (all_edge_labels == class_idx)
        if mask.sum() > 0:
            avg_local_weight = np.mean(all_edge_alphas[mask])
            avg_global_weight = 1.0 - avg_local_weight
            avg_entropy = np.mean(all_edge_entropies[mask])
            results.append({
                "Attack Class": class_name,
                "Sample Count": mask.sum(),
                "Local Stream (Inception)": avg_local_weight,
                "Global Stream (Mamba)": avg_global_weight,
                "Mean Graph Entropy": avg_entropy
            })
            
    df_results = pd.DataFrame(results)
    print("\n✅ 提取完成！各类别权重及平均结构熵：")
    print(df_results.to_string(index=False))

    csv_save_path = os.path.join(model_dir, f"gating_weights_{exp_name}.csv")
    df_results.to_csv(csv_save_path, index=False)
    plot_gating_distribution(df_results, exp_name, model_dir)

if __name__ == "__main__":
    main()

# python extract_gating_weights.py --model_path results/darknet2020_block/MILAN_NB_EXP1_BASE_dim128_seq10/20260319-200623/best_model.pth
# MILAN/results/darknet2020_block/MILAN_DEFAULT_dim128_seq10/20260317-235216/best_model.pth
# MILAN/results/cic_ids2017/MILAN_DEFAULT_dim128_seq3/20260317-233003/best_model.pth
# results/darknet2020_block/MILAN_DEFAULT_dim128_seq30/20260318-105715/best_model.pth