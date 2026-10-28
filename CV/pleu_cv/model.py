"""MoE Vision Transformer used for all CV experiments.

Every odd transformer block replaces its MLP with a top-1 routed mixture of
four experts. Parameter names are kept identical to the original
``try_moevit.py`` so that existing checkpoints load with ``strict=True``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x + self.pos_embed


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoEMLP(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=1, mlp_ratio=4):
        super().__init__()
        if top_k != 1:
            raise ValueError("MoEMLP implements top-1 routing only.")
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = int(dim * mlp_ratio)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        gate_probs = torch.softmax(self.gate(x), dim=-1)
        topk_val, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        idx = topk_idx.squeeze(-1)
        score = topk_val.squeeze(-1).unsqueeze(-1)

        out = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            mask = (idx == expert_idx).float().unsqueeze(-1)
            if mask.sum() > 0:
                out = out + mask * score * expert(x)

        dispatch_mask = F.one_hot(idx, self.num_experts).float()
        meangate = gate_probs.mean(dim=(0, 1))
        expert_usage = dispatch_mask.mean(dim=(0, 1))
        load_balance_loss = self.num_experts * (meangate * expert_usage).sum()
        return out, load_balance_loss


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, use_moe=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.use_moe = use_moe
        self.mlp = MoEMLP(dim, mlp_ratio=mlp_ratio) if use_moe else MLP(dim, mlp_ratio)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        if self.use_moe:
            out, aux_loss = self.mlp(h)
        else:
            out, aux_loss = self.mlp(h), x.new_zeros(())
        return x + out, aux_loss


class MoEVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, use_moe=(i % 2 == 1))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        aux_loss_total = 0.0
        for blk in self.blocks:
            x, aux_loss = blk(x)
            aux_loss_total = aux_loss_total + aux_loss
        logits = self.head(self.norm(x)[:, 0])
        return logits, aux_loss_total
