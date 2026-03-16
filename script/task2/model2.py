import torch
import torch.nn as nn
import timm
from dataset2 import NUM_CLASSES

# ── Config ───────────────────────────────────────────────────────────────────
CFG = {
    "backbone":   "mobilevit_xxs",
    "pretrained": True,
    "num_slices": 64,     
    "embed_dim":  320,      
    "num_classes": NUM_CLASSES,
    "dropout":    0.3,
}


class SliceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG["backbone"],
            pretrained=CFG["pretrained"],
            num_classes=0,
        )
        w = self.backbone.stem.conv.weight.data  # (out, 3, k, k)
        self.backbone.stem.conv.weight = nn.Parameter(w.mean(dim=1, keepdim=True))
        self.backbone.stem.conv.in_channels = 1

    def forward(self, x):
        return self.backbone(x)  # (B, embed_dim)

class SliceTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x: (B, S, D)
        x = self.encoder(x)
        return x.mean(dim=1)



class SliceAggregator(nn.Module):
    """Attention-weighted pooling — learns which slices matter most."""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, num_slices, embed_dim)
        w = torch.softmax(self.attn(x), dim=1)  # (B, num_slices, 1)
        return (x * w).sum(dim=1)               # (B, embed_dim)


class CTViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = SliceEncoder()
        # self.aggregator = SliceAggregator(CFG["embed_dim"])
        self.aggregator = SliceTransformer(CFG['embed_dim'])
        self.head = nn.Sequential(
            nn.LayerNorm(CFG["embed_dim"]),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(CFG["embed_dim"], CFG["num_classes"]),
        )

    def forward(self, x):
        # x: (B, 1, num_slices, H, W)
        B, C, S, H, W = x.shape
        x     = x.permute(0, 2, 1, 3, 4).reshape(B * S, C, H, W)
        feats = self.encoder(x).view(B, S, -1)
        return self.head(self.aggregator(feats))  # (B, num_classes)


# if __name__ == "__main__":
#     model = CTViT()
#     total = sum(p.numel() for p in model.parameters()) / 1e6
#     print(f"Parameters: {total:.2f}M   Classes: {CFG['num_classes']}")
#     x   = torch.randn(2, 1, CFG["num_slices"], 224, 224)
#     out = model(x)
#     print(f"Input: {x.shape}  Output: {out.shape}")