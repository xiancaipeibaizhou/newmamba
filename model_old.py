import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import softmax, degree
import math

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print(
        "Warning: mamba_ssm is not installed. MambaTemporalStream will not work. Please run: pip install mamba-ssm causal-conv1d")


# ==========================================
# 0. 基础组件: DropPath (随机深度)
# ==========================================
class DropPath(nn.Module):
    """Stochastic Depth: 在训练时随机丢弃残差路径，增强泛化能力"""

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ==========================================
# 优化版组件: 高速边到边注意力 (Fast Edge-to-Edge Attention)
# 复杂度从 O(E^2) 降至 O(E)，专为海量网络流的实时吞吐设计
# ==========================================
class FastEdgeToEdgeAttention(nn.Module):
    def __init__(self, edge_dim, heads=4, dropout=0.1):
        super().__init__()
        assert edge_dim % heads == 0, "edge_dim must be divisible by heads"

        self.heads = heads
        self.head_dim = edge_dim // heads

        self.q_proj = nn.Linear(edge_dim, edge_dim)
        self.k_proj = nn.Linear(edge_dim, edge_dim)
        self.v_proj = nn.Linear(edge_dim, edge_dim)
        self.out_proj = nn.Linear(edge_dim, edge_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(edge_dim)
        self.act = nn.GELU()

    def forward(self, edge_attr):
        E, C = edge_attr.shape
        if E == 0:
            return edge_attr

        q = F.elu(self.q_proj(edge_attr).view(E, self.heads, self.head_dim)) + 1.0
        k = F.elu(self.k_proj(edge_attr).view(E, self.heads, self.head_dim)) + 1.0
        v = self.v_proj(edge_attr).view(E, self.heads, self.head_dim)

        # [H, D, D]
        kv = torch.einsum('ehd,ehm->hdm', k, v)

        # [H, D]
        k_sum = k.sum(dim=0)

        # [E, H, 1]
        z = torch.einsum('ehd,hd->eh', q, k_sum).unsqueeze(-1)

        # [E, H, D]
        num = torch.einsum('ehd,hdm->ehm', q, kv)

        out = num / (z + 1e-6)
        out = out.reshape(E, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        return self.norm(edge_attr + self.act(out))


# ==========================================
# 1. 新增组件: Entropy-Aware Gating (并行融合门控)
# ==========================================
class EntropyGatingUnit(nn.Module):
    """
    ✨ [MR-DID 修改] 拓扑感知自适应门控单元：
    不仅根据流的全局特征，还根据显式的全图结构熵来自动调整 Local 与 Global 流的权重。
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # 输入维度增加 1，用于接收显式的图结构熵标量
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global, x_base, graph_entropy):
        """
        Args:
            x_local:  [Batch, Time, Hidden] - 来自 Inception 流
            x_global: [Batch, Time, Hidden] - 来自 Global 流 (Mamba/Attention)
            x_base:   [Batch, Time, Hidden] - 原始输入 (用于残差连接)
            graph_entropy: [Batch, 1] - 拓扑结构熵先验
        """
        # [Batch, Hidden*2]
        flow_embedding = torch.cat([x_base.mean(dim=1), x_base.max(dim=1)[0]], dim=-1)

        # 将拓扑熵拼接进流特征中: [Batch, Hidden*2 + 1]
        gate_input = torch.cat([flow_embedding, graph_entropy], dim=-1)

        # 计算门控系数 alpha [Batch, 1, 1]
        alpha = self.gate_fc(gate_input).unsqueeze(1)

        # 加权融合 + 残差连接
        out = alpha * x_local + (1 - alpha) * x_global + x_base

        return out, alpha


# ==========================================
# 2. 核心组件: Temporal Inception 1D (Local Stream)
# ==========================================
class TemporalInception1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_set=None):
        super().__init__()
        if kernel_set is None:
            kernel_set = [1, 3, 5, 7]
        if isinstance(kernel_set, (int, float)):
            kernel_set = [int(kernel_set)]
        kernel_set = [int(k) for k in kernel_set if int(k) > 0]
        if len(kernel_set) == 0:
            kernel_set = [1, 3, 5, 7]

        self.kernel_set = list(kernel_set)
        cout_per_kernel = max(1, out_features // len(self.kernel_set))

        self.tconv = nn.ModuleList()
        for kern in self.kernel_set:
            pad = kern // 2
            self.tconv.append(
                nn.Conv1d(in_features, cout_per_kernel, kernel_size=kern, padding=pad)
            )

        cat_channels = cout_per_kernel * len(self.kernel_set)
        self.fuse = nn.Identity() if cat_channels == out_features else nn.Conv1d(cat_channels, out_features,
                                                                                 kernel_size=1)

        self.project = nn.Conv1d(in_features, out_features, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        outputs = []
        for conv in self.tconv:
            outputs.append(conv(x))
        out = self.fuse(torch.cat(outputs, dim=1))
        return self.act(out + self.project(x))


# ==========================================
# 3. 核心组件: Edge-Augmented Attention (Phase 1)
# ==========================================
class EdgeAugmentedAttention(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.dropout = dropout

        assert out_dim % heads == 0, "out_dim must be divisible by heads"

        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        self.WE = nn.Linear(edge_dim, out_dim, bias=False)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = GraphNorm(out_dim)
        self.drop_path = DropPath(drop_path)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WE.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, edge_index, edge_attr, batch=None):
        residual = x
        q = self.WQ(x).view(-1, self.heads, self.head_dim)
        k = self.WK(x).view(-1, self.heads, self.head_dim)
        v = self.WV(x).view(-1, self.heads, self.head_dim)
        e_emb = self.WE(edge_attr).view(-1, self.heads, self.head_dim)

        out = self.propagate(edge_index, q=q, k=k, v=v, e_emb=e_emb, size=None)

        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        out = self.norm(out + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, e_emb, index):
        score = (q_i * (k_j + e_emb)).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = softmax(score, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * (v_j + e_emb)


# ==========================================
# 4. 核心组件: Edge Updater (Phase 1)
# ==========================================
class EdgeUpdaterModule(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        input_dim = node_dim * 2 + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hetero_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.res_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else None
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        cat_feat = torch.cat([x[src], x[dst], edge_attr], dim=-1)

        update = self.mlp(cat_feat)
        gate = self.hetero_gate(cat_feat)

        if self.res_proj is not None:
            edge_attr = self.res_proj(edge_attr)

        return self.norm(update * gate + edge_attr)


# ==========================================
# 5. 核心组件: Linear Temporal Attention
# ==========================================
class LinearTemporalAttention(nn.Module):
    def __init__(self, feature_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        B, T, C = x.shape
        residual = x

        q = F.elu(self.q_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        k = F.elu(self.k_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim)

        kv = torch.einsum('bthd,bthe->bhde', k, v)
        z = torch.einsum('bthd,bhd->bth', q, k.sum(dim=1)).unsqueeze(-1)
        num = torch.einsum('bthd,bhde->bthe', q, kv)

        out = num / (z + 1e-6)
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        return self.norm(out + residual)


# ==========================================
# 5.5 新增核心组件: Mamba Temporal Stream
# ==========================================
class MambaTemporalStream(nn.Module):
    def __init__(self, d_model, depth=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("Please install mamba_ssm: pip install mamba-ssm causal-conv1d")

        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(depth)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(depth)
        ])

    def forward(self, x):
        for blk, norm in zip(self.blocks, self.norms):
            x = x + blk(norm(x))
        return x


# ==========================================
# 6. 完整模型: MILAN (Mamba-Enhanced Parallel Dual-Stream)
# ==========================================
class MILAN(nn.Module):
    def __init__(
            self,
            node_in,
            edge_in,
            hidden,
            num_classes,
            seq_len=10,
            heads=8,
            dropout=0.3,
            max_cl_edges=2048,
            kernels=None,
            drop_path=0.1,
            dropedge_p=0.2,
            cl_view1_dropedge_p=0.1,
            cl_view2_dropedge_p=0.2,
    ):
        super(MILAN, self).__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.max_cl_edges = max_cl_edges
        self.dropedge_p = float(dropedge_p)
        self.cl_view1_dropedge_p = float(cl_view1_dropedge_p)
        self.cl_view2_dropedge_p = float(cl_view2_dropedge_p)

        # ✨ [MR-DID 修改] 引入可学习的自适应时序衰减因子
        self.decay_factor = nn.Parameter(torch.tensor(0.8))

        # --- Encoders ---
        self.node_enc = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.LayerNorm(hidden),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.LayerNorm(hidden),
        )

        # --- Spatial Layers (Phase 1) ---
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.spatial_layers.append(nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(hidden, hidden, hidden, heads, dropout, drop_path=float(drop_path)),
                'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout),
                'edge_edge_att': FastEdgeToEdgeAttention(hidden, heads, dropout)
            }))

        # --- Temporal Layers (Phase 2 & 3: Parallel) ---
        self.tpe = nn.Embedding(seq_len, hidden)

        # Stream A: Local (Inception)
        self.stream_local = TemporalInception1D(hidden, hidden, kernel_set=kernels)

        # Stream B: Global (Mamba Temporal Stream)
        self.stream_global = MambaTemporalStream(d_model=hidden, depth=2)

        # Gating: Fusion Unit (New)
        self.gating = EntropyGatingUnit(hidden)

        # --- Contrastive Head (Phase 5) ---
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.reconstruct_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden)
        )
        # --- Classifier (Phase 4) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)
        )

    # ✨ [MR-DID 修改] 基于结构熵的丢包概率计算
    def compute_structural_entropy(self, edge_index, num_nodes):
        deg = degree(edge_index[0], num_nodes, dtype=torch.float)
        p_i = 1.0 / (deg[edge_index[0]] + 1e-6)
        entropy_edge = - p_i * torch.log(p_i + 1e-6)
        return entropy_edge

    def forward(self, graphs):
        spatial_node_feats = []
        spatial_edge_feats = []
        active_edge_indices = []
        edge_masks = []
        batch_global_ids = []
        batch_graph_entropies = []

        # === Phase 1: Spatial Evolution ===
        def _spatial_encode_one_frame(data, dropedge_p):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if hasattr(data, "batch") else None
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

            if hasattr(data, "n_id"):
                frame_global_ids = data.n_id
            elif hasattr(data, "id"):
                frame_global_ids = data.id
            else:
                frame_global_ids = torch.arange(x.size(0), device=x.device)

            # ✨ [MR-DID 修改] 计算结构熵并进行自适应剪枝
            edge_entropy = self.compute_structural_entropy(edge_index, x.size(0))
            graph_entropy_scalar = edge_entropy.mean() if edge_index.size(1) > 0 else torch.tensor(0.0, device=x.device)

            if self.training and float(dropedge_p) > 0.0 and edge_index.size(1) > 0:
                norm_entropy = (edge_entropy - edge_entropy.min()) / (edge_entropy.max() - edge_entropy.min() + 1e-6)
                keep_prob = 1.0 - (float(dropedge_p) * norm_entropy)
                random_tensor = keep_prob + torch.rand_like(keep_prob)
                edge_mask = random_tensor.floor().bool()

                edge_index_d = edge_index[:, edge_mask]
                edge_attr_d = edge_attr[edge_mask]
            else:
                edge_index_d = edge_index
                edge_attr_d = edge_attr
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

            active_edge_index = edge_index_d.clone()

            x = self.node_enc(x)
            edge_attr_d = self.edge_enc(edge_attr_d)

            for layer in self.spatial_layers:
                x = layer["node_att"](x, edge_index_d, edge_attr_d, batch)
                edge_attr_d = layer["edge_upd"](x, edge_index_d, edge_attr_d)
                edge_attr_d = layer["edge_edge_att"](edge_attr_d)

            return x, edge_attr_d, active_edge_index, edge_mask, frame_global_ids, graph_entropy_scalar

        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_feat, edge_index_active, edge_mask, frame_global_ids, g_entropy = _spatial_encode_one_frame(
                data, dropedge_p=self.dropedge_p
            )

            batch_global_ids.append(frame_global_ids)
            edge_masks.append(edge_mask)
            active_edge_indices.append(edge_index_active)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_feat)
            batch_graph_entropies.append(g_entropy)

        # === Phase 2: Dynamic Alignment (Sparse to Dense) ===
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

        # ✨ [MR-DID 修改] 引入 MR-DID 风格的自适应特征衰减
        # 对在 t 时刻未激活，但在 t-1 时刻有特征的缺失节点进行平滑保留
        decay_weight = torch.sigmoid(self.decay_factor)  # 提到循环外，稍微提升性能
        for t in range(1, self.seq_len):
            missing_nodes = ~presence_mask[:, t]
            # 加上 .clone() 防止反向传播时报 in-place 修改错误
            dense_stack[missing_nodes, t, :] = dense_stack[missing_nodes, t - 1, :].clone() * decay_weight

        # === Phase 3: Temporal Evolution (Parallel Dual-Stream) ===

        time_indices = torch.arange(self.seq_len, device=device)
        t_emb = self.tpe(time_indices).unsqueeze(0)
        x_base = dense_stack + t_emb

        x_local_in = x_base.permute(0, 2, 1)
        x_local = self.stream_local(x_local_in).permute(0, 2, 1)
        x_global = self.stream_global(x_base)

        # ✨ [MR-DID 修改] 提供拓扑宏观先验给融合门控
        graph_entropies = torch.stack(batch_graph_entropies)
        mean_graph_entropy = graph_entropies.mean().unsqueeze(0).expand(num_unique, 1)

        dense_out, alpha_scores = self.gating(x_local, x_global, x_base, mean_graph_entropy)

        # === Phase 4 & 5: Readout & Contrastive ===
        batch_preds = []
        cl_loss = torch.tensor(0.0, device=device)

        for t in range(self.seq_len):
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, t, :]

            curr_edge_index = active_edge_indices[t]
            src, dst = curr_edge_index[0], curr_edge_index[1]

            edge_rep = torch.cat([
                spatial_edge_feats[t],
                node_out_t[src],
                node_out_t[dst]
            ], dim=1)

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
                    cl_loss = F.cross_entropy(logits, labels)
                else:
                    cl_loss = torch.tensor(0.0, device=device)

        self._last_edge_masks = edge_masks
        return batch_preds, cl_loss