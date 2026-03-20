import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import softmax, degree

# 从 model.py 导入所有必要组件 (完全兼容你的 MambaShield 和 Entropy-Regulated CL)
from model import (
    DropPath,
    TemporalInception1D,
    LinearTemporalAttention,
    EdgeUpdaterModule,
    EntropyGatingUnit,
    MambaTemporalStream,
    EdgeAugmentedAttention,
    FastEdgeToEdgeAttention
)

class NormalGraphAttention(MessagePassing):
    """普通的图注意力机制（不融合 Edge Features，用于 'w/o Edge Aug' 变体）"""
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.dropout = dropout
        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = GraphNorm(out_dim)
        self.drop_path = DropPath(drop_path)
        self.act = nn.GELU()
        
    def forward(self, x, edge_index, batch=None):
        residual = x
        q = self.WQ(x).view(-1, self.heads, self.head_dim)
        k = self.WK(x).view(-1, self.heads, self.head_dim)
        v = self.WV(x).view(-1, self.heads, self.head_dim)
        out = self.propagate(edge_index, q=q, k=k, v=v, size=None)
        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        out = self.norm(out + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, index):
        score = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = softmax(score, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * v_j

# ==========================================
# 基础消融模板类 (全面集成 MR-DID 逻辑)
# ==========================================
class BaseAblationMILAN(nn.Module):
    def __init__(
        self, node_in, edge_in, hidden, num_classes,
        seq_len=10, heads=8, dropout=0.3, max_cl_edges=2048,
        kernels=None, drop_path=0.1, dropedge_p=0.2
    ):
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.max_cl_edges = max_cl_edges
        self.dropedge_p = float(dropedge_p)
        self.decay_factor = nn.Parameter(torch.tensor(0.8))
        
        self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.LayerNorm(hidden))
        
        self.tpe = nn.Embedding(seq_len, hidden)
        
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.LayerNorm(hidden * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes)
        )
        # 支持新的对比学习模块
        self.reconstruct_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden)
        )

    def compute_structural_entropy(self, edge_index, num_nodes):
        deg = degree(edge_index[0], num_nodes, dtype=torch.float)
        p_i = 1.0 / (deg[edge_index[0]] + 1e-6)
        return - p_i * torch.log(p_i + 1e-6)

    def _align_temporal_features(self, batch_global_ids, spatial_node_feats):
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device
        
        dense_stack = torch.zeros((num_unique, self.seq_len, self.hidden), device=device)
        presence_mask = torch.zeros((num_unique, self.seq_len), device=device, dtype=torch.bool)
        
        for t in range(self.seq_len):
            indices = torch.searchsorted(unique_ids, batch_global_ids[t])
            dense_stack[indices, t, :] = spatial_node_feats[t]
            presence_mask[indices, t] = True
            
        decay_weight = torch.sigmoid(self.decay_factor)
        for t in range(1, self.seq_len):
            missing_nodes = ~presence_mask[:, t]
            dense_stack[missing_nodes, t, :] = dense_stack[missing_nodes, t-1, :].clone() * decay_weight
            
        time_indices = torch.arange(self.seq_len, device=device)
        t_emb = self.tpe(time_indices).unsqueeze(0)
        x_base = dense_stack + t_emb
        return x_base, unique_ids, num_unique

    def _readout_and_classify(self, dense_out, batch_global_ids, unique_ids, active_edge_indices, spatial_edge_feats, batch_graph_entropies):
        batch_preds = []
        cl_loss = torch.tensor(0.0, device=dense_out.device)
        device = dense_out.device
        
        for t in range(self.seq_len):
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, t, :] 
            
            curr_edge_index = active_edge_indices[t]
            src, dst = curr_edge_index[0], curr_edge_index[1]
            
            edge_rep = torch.cat([spatial_edge_feats[t], node_out_t[src], node_out_t[dst]], dim=1)
            pred = self.classifier(edge_rep)
            batch_preds.append(pred)
            
            if self.training and t == self.seq_len // 2:
                edge_feat_anchor = spatial_edge_feats[t]
                if edge_feat_anchor is not None and edge_feat_anchor.size(0) > 0:
                    if edge_feat_anchor.size(0) > self.max_cl_edges:
                        perm = torch.randperm(edge_feat_anchor.size(0), device=device)[: self.max_cl_edges]
                        edge_feat_anchor = edge_feat_anchor[perm]
                        edge_rep_sampled = edge_rep[perm]
                    else:
                        edge_rep_sampled = edge_rep

                    z1 = F.normalize(self.reconstruct_head(edge_rep_sampled), dim=1)
                    z2 = F.normalize(self.proj_head(edge_feat_anchor), dim=1)
                    logits = torch.matmul(z1, z2.T) / 0.1
                    labels = torch.arange(z1.size(0), device=device)
                    base_cl_loss = F.cross_entropy(logits, labels)
                    
                    # ✨ 兼容最新的 Entropy-Regulated CL
                    current_frame_entropy = batch_graph_entropies[t]
                    dynamic_cl_scale = 1.0 + torch.tanh(current_frame_entropy)
                    cl_loss = base_cl_loss * dynamic_cl_scale
                else:
                    cl_loss = torch.tensor(0.0, device=device)

        return batch_preds, cl_loss


# ==========================================
# 变体 1: w/o Global (移除全局注意力, 仅用 Inception)
# ==========================================
class MILAN_WoGlobal(BaseAblationMILAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(self.hidden, self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3)),
                'edge_edge_att': FastEdgeToEdgeAttention(self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        self.stream_local = TemporalInception1D(self.hidden, self.hidden, kernel_set=kwargs.get('kernels'))

    def forward(self, graphs):
        spatial_node_feats, spatial_edge_feats, active_edge_indices, batch_global_ids, batch_graph_entropies = [], [], [], [], []
        edge_masks = []
        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = self.node_enc(torch.nan_to_num(data.x)), data.edge_index, self.edge_enc(torch.nan_to_num(data.edge_attr))
            
            edge_entropy = self.compute_structural_entropy(edge_index, x.size(0))
            graph_entropy_scalar = edge_entropy.mean() if edge_index.size(1) > 0 else torch.tensor(0.0, device=x.device)
            batch_graph_entropies.append(graph_entropy_scalar)
            
            if self.training and self.dropedge_p > 0.0 and edge_index.size(1) > 0:
                norm_entropy = (edge_entropy - edge_entropy.min()) / (edge_entropy.max() - edge_entropy.min() + 1e-6)
                keep_prob = 1.0 - (self.dropedge_p * norm_entropy)
                edge_mask = (keep_prob + torch.rand_like(keep_prob)).floor().bool()
                edge_index = edge_index[:, edge_mask]
                edge_attr = edge_attr[edge_mask]
            else:
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            edge_masks.append(edge_mask)
            
            for layer in self.spatial_layers:
                x = layer["node_att"](x, edge_index, edge_attr, getattr(data, 'batch', None))
                edge_attr = layer["edge_upd"](x, edge_index, edge_attr)
                edge_attr = layer["edge_edge_att"](edge_attr)
            active_edge_indices.append(edge_index)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)
            batch_global_ids.append(data.n_id if hasattr(data, "n_id") else torch.arange(x.size(0), device=x.device))

        x_base, unique_ids, num_unique = self._align_temporal_features(batch_global_ids, spatial_node_feats)
        x_local_in = x_base.permute(0, 2, 1) 
        dense_out = self.stream_local(x_local_in).permute(0, 2, 1) + x_base 
        self._last_edge_masks = edge_masks
        return self._readout_and_classify(dense_out, batch_global_ids, unique_ids, active_edge_indices, spatial_edge_feats, batch_graph_entropies)

# ==========================================
# 变体 2: w/o Local (移除局部卷积，保留 Mamba)
# ==========================================
class MILAN_WoLocal(BaseAblationMILAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(self.hidden, self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3)),
                'edge_edge_att': FastEdgeToEdgeAttention(self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        self.stream_global = MambaTemporalStream(d_model=self.hidden, depth=2)

    def forward(self, graphs):
        spatial_node_feats, spatial_edge_feats, active_edge_indices, batch_global_ids, batch_graph_entropies = [], [], [], [], []
        edge_masks = []
        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = self.node_enc(torch.nan_to_num(data.x)), data.edge_index, self.edge_enc(torch.nan_to_num(data.edge_attr))
            
            edge_entropy = self.compute_structural_entropy(edge_index, x.size(0))
            graph_entropy_scalar = edge_entropy.mean() if edge_index.size(1) > 0 else torch.tensor(0.0, device=x.device)
            batch_graph_entropies.append(graph_entropy_scalar)
            
            if self.training and self.dropedge_p > 0.0 and edge_index.size(1) > 0:
                norm_entropy = (edge_entropy - edge_entropy.min()) / (edge_entropy.max() - edge_entropy.min() + 1e-6)
                keep_prob = 1.0 - (self.dropedge_p * norm_entropy)
                edge_mask = (keep_prob + torch.rand_like(keep_prob)).floor().bool()
                edge_index = edge_index[:, edge_mask]
                edge_attr = edge_attr[edge_mask]
            else:
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            edge_masks.append(edge_mask)
            
            for layer in self.spatial_layers:
                x = layer["node_att"](x, edge_index, edge_attr, getattr(data, 'batch', None))
                edge_attr = layer["edge_upd"](x, edge_index, edge_attr)
                edge_attr = layer["edge_edge_att"](edge_attr)
            active_edge_indices.append(edge_index)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)
            batch_global_ids.append(data.n_id if hasattr(data, "n_id") else torch.arange(x.size(0), device=x.device))

        x_base, unique_ids, num_unique = self._align_temporal_features(batch_global_ids, spatial_node_feats)
        
        # 传入 MambaShield 所需的 graph_entropy
        mean_graph_entropy = torch.stack(batch_graph_entropies).mean().unsqueeze(0).expand(num_unique, 1)
        dense_out = self.stream_global(x_base, mean_graph_entropy) + x_base 
        
        self._last_edge_masks = edge_masks
        return self._readout_and_classify(dense_out, batch_global_ids, unique_ids, active_edge_indices, spatial_edge_feats, batch_graph_entropies)


# ==========================================
# 变体 3: Linear Transformer (替代 Mamba)
# ==========================================
class MILAN_LinearTransformer(BaseAblationMILAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(self.hidden, self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3)),
                'edge_edge_att': FastEdgeToEdgeAttention(self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        
        self.stream_local = TemporalInception1D(self.hidden, self.hidden, kernel_set=kwargs.get('kernels'))
        self.stream_global = LinearTemporalAttention(self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3))
        self.gating = EntropyGatingUnit(self.hidden)

    def forward(self, graphs):
        spatial_node_feats, spatial_edge_feats, active_edge_indices, batch_global_ids, batch_graph_entropies = [], [], [], [], []
        edge_masks = []
        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = self.node_enc(torch.nan_to_num(data.x)), data.edge_index, self.edge_enc(torch.nan_to_num(data.edge_attr))
            
            edge_entropy = self.compute_structural_entropy(edge_index, x.size(0))
            graph_entropy_scalar = edge_entropy.mean() if edge_index.size(1) > 0 else torch.tensor(0.0, device=x.device)
            batch_graph_entropies.append(graph_entropy_scalar)
            
            if self.training and self.dropedge_p > 0.0 and edge_index.size(1) > 0:
                norm_entropy = (edge_entropy - edge_entropy.min()) / (edge_entropy.max() - edge_entropy.min() + 1e-6)
                keep_prob = 1.0 - (self.dropedge_p * norm_entropy)
                edge_mask = (keep_prob + torch.rand_like(keep_prob)).floor().bool()
                edge_index = edge_index[:, edge_mask]
                edge_attr = edge_attr[edge_mask]
            else:
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            edge_masks.append(edge_mask)
            
            for layer in self.spatial_layers:
                x = layer["node_att"](x, edge_index, edge_attr, getattr(data, 'batch', None))
                edge_attr = layer["edge_upd"](x, edge_index, edge_attr)
                edge_attr = layer["edge_edge_att"](edge_attr)
            active_edge_indices.append(edge_index)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)
            batch_global_ids.append(data.n_id if hasattr(data, "n_id") else torch.arange(x.size(0), device=x.device))

        x_base, unique_ids, num_unique = self._align_temporal_features(batch_global_ids, spatial_node_feats)
        
        x_local_in = x_base.permute(0, 2, 1) 
        x_local = self.stream_local(x_local_in).permute(0, 2, 1)
        
        # ✨ LinearAttention 不接受 graph_entropy 参数，仅传入 x_base
        x_global = self.stream_global(x_base)
        
        mean_graph_entropy = torch.stack(batch_graph_entropies).mean().unsqueeze(0).expand(num_unique, 1)
        dense_out, _ = self.gating(x_local, x_global, x_base, mean_graph_entropy)
        
        self._last_edge_masks = edge_masks
        return self._readout_and_classify(dense_out, batch_global_ids, unique_ids, active_edge_indices, spatial_edge_feats, batch_graph_entropies)

# ==========================================
# 变体 4: w/o Gating (直接相加融合)
# ==========================================
class MILAN_WoGating(MILAN_LinearTransformer):
    def forward(self, graphs):
        # 复制上述相同的前向传播抽取部分 (省略重复，直接到对齐部分)
        spatial_node_feats, spatial_edge_feats, active_edge_indices, batch_global_ids, batch_graph_entropies = [], [], [], [], []
        edge_masks = []
        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = self.node_enc(torch.nan_to_num(data.x)), data.edge_index, self.edge_enc(torch.nan_to_num(data.edge_attr))
            edge_entropy = self.compute_structural_entropy(edge_index, x.size(0))
            batch_graph_entropies.append(edge_entropy.mean() if edge_index.size(1) > 0 else torch.tensor(0.0, device=x.device))
            if self.training and self.dropedge_p > 0.0 and edge_index.size(1) > 0:
                norm_entropy = (edge_entropy - edge_entropy.min()) / (edge_entropy.max() - edge_entropy.min() + 1e-6)
                keep_prob = 1.0 - (self.dropedge_p * norm_entropy)
                edge_mask = (keep_prob + torch.rand_like(keep_prob)).floor().bool()
                edge_index = edge_index[:, edge_mask]
                edge_attr = edge_attr[edge_mask]
            else:
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            edge_masks.append(edge_mask)
            for layer in self.spatial_layers:
                x = layer["node_att"](x, edge_index, edge_attr, getattr(data, 'batch', None))
                edge_attr = layer["edge_upd"](x, edge_index, edge_attr)
                edge_attr = layer["edge_edge_att"](edge_attr)
            active_edge_indices.append(edge_index)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)
            batch_global_ids.append(data.n_id if hasattr(data, "n_id") else torch.arange(x.size(0), device=x.device))

        x_base, unique_ids, num_unique = self._align_temporal_features(batch_global_ids, spatial_node_feats)
        x_local_in = x_base.permute(0, 2, 1) 
        x_local = self.stream_local(x_local_in).permute(0, 2, 1)
        x_global = self.stream_global(x_base) # 同样，不需要 entropy
        
        # 移除了 Gating，直接相加
        dense_out = x_local + x_global + x_base 
        self._last_edge_masks = edge_masks
        return self._readout_and_classify(dense_out, batch_global_ids, unique_ids, active_edge_indices, spatial_edge_feats, batch_graph_entropies)

# ==========================================
# 变体 5: w/o Edge Aug (移除边特征增强注意力)
# ==========================================
class MILAN_WoEdgeAug(BaseAblationMILAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': NormalGraphAttention(self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3)),
                'edge_edge_att': FastEdgeToEdgeAttention(self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        self.stream_local = TemporalInception1D(self.hidden, self.hidden, kernel_set=kwargs.get('kernels'))
        self.stream_global = MambaTemporalStream(d_model=self.hidden, depth=2)
        self.gating = EntropyGatingUnit(self.hidden)

    def forward(self, graphs):
        spatial_node_feats, spatial_edge_feats, active_edge_indices, batch_global_ids, batch_graph_entropies = [], [], [], [], []
        edge_masks = []
        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = self.node_enc(torch.nan_to_num(data.x)), data.edge_index, self.edge_enc(torch.nan_to_num(data.edge_attr))
            
            edge_entropy = self.compute_structural_entropy(edge_index, x.size(0))
            graph_entropy_scalar = edge_entropy.mean() if edge_index.size(1) > 0 else torch.tensor(0.0, device=x.device)
            batch_graph_entropies.append(graph_entropy_scalar)
            
            if self.training and self.dropedge_p > 0.0 and edge_index.size(1) > 0:
                norm_entropy = (edge_entropy - edge_entropy.min()) / (edge_entropy.max() - edge_entropy.min() + 1e-6)
                keep_prob = 1.0 - (self.dropedge_p * norm_entropy)
                edge_mask = (keep_prob + torch.rand_like(keep_prob)).floor().bool()
                edge_index = edge_index[:, edge_mask]
                edge_attr = edge_attr[edge_mask]
            else:
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            edge_masks.append(edge_mask)
            
            for layer in self.spatial_layers:
                # 传入普通注意力，没有 edge_attr
                x = layer["node_att"](x, edge_index, getattr(data, 'batch', None))
                edge_attr = layer["edge_upd"](x, edge_index, edge_attr)
                edge_attr = layer["edge_edge_att"](edge_attr)

            active_edge_indices.append(edge_index)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)
            batch_global_ids.append(data.n_id if hasattr(data, "n_id") else torch.arange(x.size(0), device=x.device))

        x_base, unique_ids, num_unique = self._align_temporal_features(batch_global_ids, spatial_node_feats)
        x_local_in = x_base.permute(0, 2, 1) 
        x_local = self.stream_local(x_local_in).permute(0, 2, 1)
        
        mean_graph_entropy = torch.stack(batch_graph_entropies).mean().unsqueeze(0).expand(num_unique, 1)
        
        # Mamba 模块需要 mean_graph_entropy 开启 Shield
        x_global = self.stream_global(x_base, mean_graph_entropy)
        
        dense_out, _ = self.gating(x_local, x_global, x_base, mean_graph_entropy)
        
        self._last_edge_masks = edge_masks
        return self._readout_and_classify(dense_out, batch_global_ids, unique_ids, active_edge_indices, spatial_edge_feats, batch_graph_entropies)

# ==========================================
# 变体 6: Standard Transformer
# ==========================================
class MILAN_StandardTransformer(MILAN_LinearTransformer):
    def __init__(self, *args, **kwargs):
        # 初始化与 Linear 完全一致，只需要把 stream_global 换成 Transformer
        super(MILAN_LinearTransformer, self).__init__(*args, **kwargs)
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(self.hidden, self.hidden, self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3), drop_path=kwargs.get('drop_path', 0.1)),
                'edge_upd': EdgeUpdaterModule(self.hidden, self.hidden, self.hidden, kwargs.get('dropout', 0.3)),
                'edge_edge_att': FastEdgeToEdgeAttention(self.hidden, kwargs.get('heads', 8), kwargs.get('dropout', 0.3))
            }) for _ in range(self.num_layers)
        ])
        self.stream_local = TemporalInception1D(self.hidden, self.hidden, kernel_set=kwargs.get('kernels'))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=kwargs.get('heads', 8), 
            dim_feedforward=self.hidden * 2, dropout=kwargs.get('dropout', 0.3),
            batch_first=True, activation='gelu'
        )
        self.stream_global = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.gating = EntropyGatingUnit(self.hidden)
    
    # Forward 完全继承自 MILAN_LinearTransformer 即可，因为 TransformerEncoder 也不接受 graph_entropy 参数