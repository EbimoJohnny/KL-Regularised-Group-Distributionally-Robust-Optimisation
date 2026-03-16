

import torch
import torch.nn as nn
import timm

# ── Config ───────────────────────────────────────────────────────────────────
CFG = {
    "backbone":     "mobilevit_xxs",  
    "pretrained":   True,
    "num_slices":   64,                
    "embed_dim":    320,             
    "num_classes":  2,
    "dropout":      0.3,
}


class SliceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG["backbone"],
            pretrained=CFG["pretrained"],
            num_classes=0,          # remove classification head and returns features
        )
        # patch: accepts 1-channel input by averaging pretrained conv1 weights
        w = self.backbone.stem.conv.weight.data  # (out, 3, k, k)
        self.backbone.stem.conv.weight = nn.Parameter(w.mean(dim=1, keepdim=True))
        self.backbone.stem.conv.in_channels = 1

    def forward(self, x):
        return self.backbone(x)   


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
    """Weighted pooling across slice dimension — learns which slices matter most."""
    def __init__(self, embed_dim, num_slices):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, num_slices, embed_dim)
        w = self.attn(x)                    
        w = torch.softmax(w, dim=1)        
        return (x * w).sum(dim=1)      


# ── Full Model ────────────────────────────────────────────────────────────────
class CovidViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = SliceEncoder()
        # self.aggregator = SliceAggregator(CFG["embed_dim"], CFG["num_slices"])
        self.aggregator = SliceTransformer(CFG["embed_dim"])
        self.head = nn.Sequential(
            nn.LayerNorm(CFG["embed_dim"]),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(CFG["embed_dim"], CFG["num_classes"]),
        )

    def forward(self, x):
        # x: (B, 1, num_slices, H, W)
        B, C, S, H, W = x.shape

        # encode each slice independently
        x = x.permute(0, 2, 1, 3, 4).reshape(B * S, C, H, W)  # (B*S, 1, H, W)
        feats = self.encoder(x)                                
        feats = feats.view(B, S, -1)                             # (B, S, embed_dim)

        # aggregate across slices
        pooled = self.aggregator(feats)   # (B, embed_dim)

        return self.head(pooled)          # (B, num_classes)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CovidViT()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.2f}M")

    x = torch.randn(2, 1, CFG["num_slices"], 224, 224)
    out = model(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")   # (2, 2)
    