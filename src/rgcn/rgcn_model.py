"""
R-GCN 实现 (按 Schlichtkrull et al., ESWC 2018).

只依赖 torch, 不依赖 DGL/PyG, 用 torch.scatter_add_ 做消息传递,
适合 small/medium 异构图 (本任务 1155 节点, 10K 边).

核心公式 (第 l 层):

    h_v^(l+1) = σ( W_self^(l) h_v^(l)
                 + Σ_r∈R Σ_u∈N_r(v) (w_{u,v,r} / c_{v,r}) · W_r^(l) h_u^(l) )

其中:
  - W_r 用 basis decomposition: W_r = Σ_{b=1..B} a_{rb} · V_b
    R 个关系仅共享 B 个基矩阵 (B << R 时可极大压缩参数)
  - w_{u,v,r}: 边权 (log(1+count), 编码事件频次)
  - c_{v,r} = Σ_u w_{u,v,r} : 行归一化, 防止度数大的节点压过别人

实现细节:
  * 节点是 flat-index (0..N-1), 不区分 ntype, 用一份 X∈R^{N×D}
  * 每条关系给出 (src, dst, weight); 我们额外自动添加反向边
    rel_inv -> 用单独的可学习 W_{r_inv}, 让 PC/file/url 节点也能收到信息
  * self-loop 单独用 W_self
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 关系层 (一个 RGCN block)
# ---------------------------------------------------------------------------
class RGCNLayer(nn.Module):
    """
    一层 R-GCN. 输入: X ∈ [N, in_dim], 边集 edges {rel: (src, dst, w)}.
    输出: H ∈ [N, out_dim].

    Args:
        in_dim       : 输入维度
        out_dim      : 输出维度
        num_relations: 关系数 R (含反向); 与 edges dict 的 key 数一致
        num_bases    : basis decomposition 的 B; B<=num_relations
        dropout      : 输出前 dropout
        activation   : 激活函数 (None 表示最后一层不激活)
        self_loop    : 是否带自环 W_self
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        num_bases: int = 4,
        dropout: float = 0.2,
        activation=F.relu,
        self_loop: bool = True,
    ):
        super().__init__()
        assert 1 <= num_bases <= num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.activation = activation
        self.self_loop = self_loop

        # B 个基矩阵 V_b ∈ R^{in×out}
        self.bases = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        # 关系系数 a_{r,b} ∈ R^{R×B}
        self.coeff = nn.Parameter(torch.empty(num_relations, num_bases))
        # self-loop
        if self_loop:
            self.W_self = nn.Parameter(torch.empty(in_dim, out_dim))
        # bias
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

        # 初始化 (Xavier uniform; 与原论文相同思路)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeff)
        if self_loop:
            nn.init.xavier_uniform_(self.W_self)

    def relation_weight(self) -> torch.Tensor:
        """合成 R 个关系矩阵: W = coeff @ bases -> [R, in, out]"""
        # (R, B) x (B, in*out) -> (R, in*out)
        flat = self.coeff @ self.bases.view(self.num_bases, -1)
        return flat.view(self.num_relations, self.in_dim, self.out_dim)

    def forward(
        self,
        x: torch.Tensor,                                    # [N, in_dim]
        edges: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        # 长度 = num_relations 的列表, 每项 (src, dst, weight)
    ) -> torch.Tensor:
        assert len(edges) == self.num_relations, \
            f'edges len={len(edges)} != num_relations={self.num_relations}'
        N = x.size(0)
        W = self.relation_weight()                          # [R, in, out]

        out = torch.zeros(N, self.out_dim, device=x.device, dtype=x.dtype)

        # 预算 dst 的"加权度": 用于行归一化 c_{v,r}, 跨关系直接累加
        deg = torch.zeros(N, device=x.device, dtype=x.dtype)
        for r, (src, dst, w) in enumerate(edges):
            if src.numel() == 0:
                continue
            deg.scatter_add_(0, dst, w)
        deg = torch.clamp(deg, min=1e-6)

        # 消息传递: 每个关系
        for r, (src, dst, w) in enumerate(edges):
            if src.numel() == 0:
                continue
            # 1) 用关系 r 的 W_r 把源节点先投影到 out_dim
            #    msg_e = w · (X[src] @ W_r)
            x_src = x[src]                                  # [E_r, in]
            msg = x_src @ W[r]                              # [E_r, out]
            msg = msg * w.unsqueeze(-1)                     # 边权
            # 2) scatter_add 到 dst
            out.index_add_(0, dst, msg)

        # 行归一化 (按 dst 的总加权度)
        out = out / deg.unsqueeze(-1)

        # self-loop
        if self.self_loop:
            out = out + x @ self.W_self

        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# 全模型: 节点 embedding + L 层 R-GCN + 二分类头
# ---------------------------------------------------------------------------
class RGCNClassifier(nn.Module):
    """
    R-GCN 节点分类器 (user 节点 二分类).

    异构节点特征处理:
      - user 节点: 6 维数值特征 -> Linear -> emb_dim
      - pc / file_type / url_cat 节点: nn.Embedding(num, emb_dim)
      四种 ntype 的初始表示拼到一个 [N, emb_dim] 张量上, 然后过 R-GCN.

    Args:
        node_offsets : {ntype: (start, end)}  flat-index 切片
        user_feat_dim: 6
        emb_dim      : 隐藏维度 (统一 hidden)
        num_relations: 关系数 (含反向, 通常 = 2 * R_orig)
        num_layers   : R-GCN 层数 (推荐 2)
        num_bases    : basis decomposition B
        dropout      : 层间 dropout
    """

    def __init__(
        self,
        node_offsets: Dict[str, Tuple[int, int]],
        user_feat_dim: int,
        emb_dim: int,
        num_relations: int,
        num_layers: int = 2,
        num_bases: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_offsets = node_offsets
        self.emb_dim = emb_dim

        # user 节点: 把 6 维特征投影到 emb_dim
        self.user_proj = nn.Linear(user_feat_dim, emb_dim)
        # 其它三类节点: 各自一个 Embedding
        self.pc_emb        = nn.Embedding(self._size('pc'),        emb_dim)
        self.file_emb      = nn.Embedding(self._size('file_type'), emb_dim)
        self.url_emb       = nn.Embedding(self._size('url_cat'),   emb_dim)

        # R-GCN 层栈
        self.layers = nn.ModuleList()
        for li in range(num_layers):
            self.layers.append(RGCNLayer(
                in_dim=emb_dim, out_dim=emb_dim,
                num_relations=num_relations, num_bases=num_bases,
                dropout=dropout,
                activation=F.relu if li < num_layers - 1 else None,
                self_loop=True,
            ))

        # 二分类 head (只用在 user 节点上)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1),
        )

        # 初始化 embedding (xavier)
        for emb in [self.pc_emb, self.file_emb, self.url_emb]:
            nn.init.xavier_uniform_(emb.weight)

    def _size(self, ntype: str) -> int:
        s, e = self.node_offsets[ntype]
        return e - s

    def initial_features(self, user_feats: torch.Tensor) -> torch.Tensor:
        """组装 X ∈ [N, emb_dim], 按 flat-index 顺序拼接."""
        device = user_feats.device
        N = max(e for _, e in self.node_offsets.values())
        x = torch.zeros(N, self.emb_dim, device=device)
        # user
        us, ue = self.node_offsets['user']
        x[us:ue] = self.user_proj(user_feats)
        # pc
        ps, pe = self.node_offsets['pc']
        x[ps:pe] = self.pc_emb.weight
        # file
        fs, fe = self.node_offsets['file_type']
        x[fs:fe] = self.file_emb.weight
        # url
        hs, he = self.node_offsets['url_cat']
        x[hs:he] = self.url_emb.weight
        return x

    def encode(
        self,
        user_feats: torch.Tensor,
        edges: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        x = self.initial_features(user_feats)
        for layer in self.layers:
            x = layer(x, edges)
        return x   # [N, emb_dim]

    def forward(
        self,
        user_feats: torch.Tensor,
        edges: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """返回 user 节点的 logits, shape [N_user]."""
        h = self.encode(user_feats, edges)
        us, ue = self.node_offsets['user']
        logits = self.classifier(h[us:ue]).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# 反向关系工具: 给 R-GCN 添加反向边, 让信息双向流动
# ---------------------------------------------------------------------------
def add_reverse_edges(
    edges_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    relation_names: List[str],
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], List[str]]:
    """
    把 {rel: (src, dst, w)} 字典扩展成有反向边的 edges list,
    顺序 [r1, r1_rev, r2, r2_rev, ...], 长度 = 2 * R.

    Returns:
        edges_list  : List[(src, dst, w)] 共 2R 项, 与 num_relations 对齐
        rel_names_2 : List[str] 共 2R 项 (用于 debug/可视化)
    """
    out_edges = []
    out_names = []
    for r in relation_names:
        src, dst, w = edges_dict[r]
        out_edges.append((src, dst, w))
        out_names.append(r)
        out_edges.append((dst, src, w))
        out_names.append(r + '_rev')
    return out_edges, out_names
