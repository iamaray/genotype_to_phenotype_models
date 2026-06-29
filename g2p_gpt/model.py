import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dff: int):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.ffnn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        )

    def forward(self, x: torch.Tensor):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        ffnn_out = self.ffnn(x_norm)
        x = x + self.dropout2(ffnn_out)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dff: int):
        super().__init__()

        self.atten1 = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True)
        self.atten2 = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        self.ffnn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        )

    def forward(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
        # tgt_mask: torch.Tensor | None = None
    ):
        # y: decoder input embeddings, shape (B, T_tgt, d_model)
        # z: encoder output / memory, shape (B, T_src, d_model)

        y_norm = self.norm1(y)
        attn_out, _ = self.atten1(
            y_norm, y_norm, y_norm,
            need_weights=False,
            # attn_mask=tgt_mask,
        )
        y = y + self.dropout1(attn_out)

        y_norm = self.norm2(y)
        attn_out, _ = self.atten2(
            y_norm, z, z,
            need_weights=False,
        )
        y = y + self.dropout2(attn_out)

        y_norm = self.norm3(y)
        ffnn_out = self.ffnn(y_norm)
        y = y + self.dropout3(ffnn_out)

        return y


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dff: int,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6
    ):

        super().__init__()

        self.d_model = d_model

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.encoders = nn.ModuleList(
            [TransformerEncoder(num_heads, d_model, dff)
             for _ in range(num_enc_layers)])

        self.decoders = nn.ModuleList(
            [TransformerDecoder(num_heads, d_model, dff)
             for _ in range(num_dec_layers)])

        self.ffnns = nn.ModuleDict({
            str(k): nn.Linear(in_features=d_model, out_features=4 ** k)
            for k in (3, 4, 5, 6)
        })

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            k: int):
        # x input embeddings
        # y right-shifted output embeddings

        z = x.clone()

        for enc in self.encoders:
            z = enc(z)
        for dec in self.decoders:
            y = dec(y, z)

        y = self.ffnns[str(k)](y)
        return y

    def encode(self):
        pass

    def decode(self):
        pass

    def predict(self):
        pass
