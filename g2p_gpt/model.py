import torch
import torch.nn as nn

from .helpers import make_causal_mask, sinusoidal_positional_encoding


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
        tgt_mask: torch.Tensor | None = None
    ):
        # y: decoder input embeddings, shape (B, T_tgt, d_model)
        # z: encoder output / memory, shape (B, T_src, d_model)

        y_norm = self.norm1(y)
        attn_out, _ = self.atten1(
            y_norm, y_norm, y_norm,
            attn_mask=tgt_mask,
            need_weights=False,
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
        num_dec_layers: int = 6,
        vocab_size: int = 64
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

        self.ffnn = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x input embeddings
        # y shifted-right output embeddings

        pos_enc = sinusoidal_positional_encoding(
            y.shape[1], self.d_model, self.device).unsqueeze(0)

        # print("pos enc shape", pos_enc.shape)
        # print("x shape", x.shape)

        z = x.clone() + pos_enc
        y = y + pos_enc

        causal_mask = make_causal_mask(y.shape[1])

        for enc in self.encoders:
            z = enc(z)
        for dec in self.decoders:
            y = dec(y, z, causal_mask)

        y = self.ffnn(y)
        return y

    def predict(self):
        pass


class G2P_GPT(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 dff: int,
                 num_enc_layers: int = 6,
                 num_dec_layers: int = 6):
        super().__init__()

        self.emb3 = nn.Embedding(64, d_model)
        self.emb4 = nn.Embedding(256, d_model)
        self.emb5 = nn.Embedding(1024, d_model)
        self.emb6 = nn.Embedding(4096, d_model)

        self.transformer_3mer = Transformer(
            num_heads, d_model, dff,
            num_enc_layers, num_dec_layers, vocab_size=64)
        self.transformer_4mer = Transformer(
            num_heads, d_model, dff,
            num_enc_layers, num_dec_layers, vocab_size=256)
        self.transformer_5mer = Transformer(
            num_heads, d_model, dff,
            num_enc_layers, num_dec_layers, vocab_size=1024)
        self.transformer_6mer = Transformer(
            num_heads, d_model, dff,
            num_enc_layers, num_dec_layers, vocab_size=4096)

    def forward(self, toks: torch.Tensor):
        x3 = tok_to_emb(toks, self.emb3)
        x4 = tok_to_emb(toks, self.emb4)
        x5 = tok_to_emb(toks, self.emb5)
        x6 = tok_to_emb(toks, self.emb6)

        out3 = self.transformer_3mer(x3[:, :-1, :], x3[:, 1:, :])
        out4 = self.transformer_4mer(x4[:, :-1, :], x4[:, 1:, :])
        out5 = self.transformer_5mer(x5[:, :-1, :], x5[:, 1:, :])
        out6 = self.transformer_6mer(x6[:, :-1, :], x6[:, 1:, :])

        return {
            "3mer": out3,
            "4mer": out4,
            "5mer": out5,
            "6mer": out6
        }

    def predict(self):
        pass
