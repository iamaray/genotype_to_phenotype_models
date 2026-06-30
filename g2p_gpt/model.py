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

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm,
            need_weights=False,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        ffnn_out = self.ffnn(x_norm)
        x = x + self.dropout2(ffnn_out)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dff: int,
        causal: bool = False,
    ):
        super().__init__()

        self.causal = causal

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
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ):
        # y: decoder input embeddings, shape (B, T_tgt, d_model)
        # z: encoder output / memory, shape (B, T_src, d_model)
        # bool masks follow torch.nn.MultiheadAttention semantics:
        # True entries are masked out.
        if self.causal and tgt_mask is None:
            raise ValueError(
                "causal=True requires tgt_mask to be passed to forward()."
            )

        y_norm = self.norm1(y)
        attn_out, _ = self.atten1(
            y_norm, y_norm, y_norm,
            need_weights=False,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        y = y + self.dropout1(attn_out)

        y_norm = self.norm2(y)
        attn_out, _ = self.atten2(
            y_norm, z, z,
            need_weights=False,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        y = y + self.dropout2(attn_out)

        y_norm = self.norm3(y)
        ffnn_out = self.ffnn(y_norm)
        y = y + self.dropout3(ffnn_out)

        return y


class Transformer(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_heads: int,
        d_model: int,
        dff: int,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        k_choices: list = [3, 4, 5, 6],
        causal: bool = False,
    ):

        super().__init__()

        self.model_id = model_id
        self.d_model = d_model
        self.causal = causal

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.encoders = nn.ModuleList(
            [TransformerEncoder(num_heads, d_model, dff)
             for _ in range(num_enc_layers)])

        self.decoders = nn.ModuleList(
            [TransformerDecoder(num_heads, d_model, dff, causal=causal)
             for _ in range(num_dec_layers)])

        self.ffnns = nn.ModuleDict({
            str(k): nn.Linear(in_features=d_model, out_features=4 ** k)
            for k in k_choices
        })

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            k: int,
            src_mask: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            src_key_padding_mask: torch.Tensor | None = None,
            tgt_key_padding_mask: torch.Tensor | None = None,
            memory_key_padding_mask: torch.Tensor | None = None):
        # x input embeddings
        # y output embeddings
        if self.causal and tgt_mask is None:
            raise ValueError(
                "causal=True requires tgt_mask to be passed to forward()."
            )
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask

        z = x.clone()

        for enc in self.encoders:
            z = enc(
                z,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        for dec in self.decoders:
            y = dec(
                y,
                z,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        y = self.ffnns[str(k)](y)
        return y

    def encode(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ):
        self.train(False)
        for enc in self.encoders:
            x = enc(
                x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        return x

    def decode(self):
        pass
