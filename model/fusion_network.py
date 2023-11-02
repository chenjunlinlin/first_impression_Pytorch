from turtle import forward
from sympy import false
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.2):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, querry):
        x = self.norm(x)

        kv = self.to_kv(x).chunk(2, dim=-1)
        q = self.to_q(querry)
        k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads,
                          dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, querry):
        for attn, ff in self.layers:
            x = attn(x, querry) + x
            x = ff(x) + x

        return self.norm(x)


class Fusion(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, cfg, *args, **kwargs) -> None:
        super(Fusion, self).__init__(*args, **kwargs)
        self.pos_embeding_img = nn.Parameter(torch.randn(1, cfg.N, 512))
        self.pos_embeding_audio = nn.Parameter(torch.randn(1, cfg.N, 512))
        # self.embeding_img = nn.Linear(cfg.dim_img, 512)
        # self.embeding_audio = nn.Linear(cfg.dim_audio, 512)
        self.fusion_layer = Transformer(dim=dim,
                                        depth=depth,
                                        heads=heads,
                                        dim_head=dim_head,
                                        mlp_dim=mlp_dim)

    def forward(self, x, q):
        X = x + self.pos_embeding_img
        Q = q + self.pos_embeding_audio
        X = self.fusion_layer(X, Q)
        return X + x
