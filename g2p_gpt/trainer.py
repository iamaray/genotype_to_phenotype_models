import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..helpers import make_causal_mask, sinusoidal_positional_encoding, tok_to_emb


def train(
        params: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        lr: float = 3e-4,
        pad_token_id: int | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """ Trains k-mer embeddings on their respective next k-mer token 
        prediction tasks via vanilla transformer models.

    Args:
        params (dict): _description_
        train_loader (DataLoader): _description_
        val_loader (DataLoader): _description_
        num_epochs (int): _description_
        lr (float, optional): _description_. Defaults to 3e-4.
        pad_token_id (int | None, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cuda"iftorch.cuda.is_available()else"cpu".
    """

    device = torch.device(device)

    d_model = params['d_model']
    num_heads = params['num_heads']
    dff = params['dff']
    num_enc_layers = params['num_enc_layers']
    num_dec_layers = params['num_dec_layers']

    emb3 = nn.Embedding(64, d_model)
    emb4 = nn.Embedding(256, d_model)
    emb5 = nn.Embedding(1024, d_model)
    emb6 = nn.Embedding(4096, d_model)

    transformer_3mer = Transformer(
        num_heads, d_model, dff,
        num_enc_layers, num_dec_layers, vocab_size=64).to(device)
    transformer_4mer = Transformer(
        num_heads, d_model, dff,
        num_enc_layers, num_dec_layers, vocab_size=256).to(device)
    transformer_5mer = Transformer(
        num_heads, d_model, dff,
        num_enc_layers, num_dec_layers, vocab_size=1024).to(device)
    transformer_6mer = Transformer(
        num_heads, d_model, dff,
        num_enc_layers, num_dec_layers, vocab_size=4096).to(device)

    model_param_list = list(transformer_3mer.parameters()) + list(transformer_3mer.parameters()) + list(
        transformer_4mer.parameters()) + list(transformer_5mer.parameters()) + list(transformer_6mer.parameters())

    embed_param_list = list(model.parameters()) + list(emb3.parameters()) + list(
        emb4.parameters()) + list(emb5.parameters()) + list(emb6.parameters())

    optimizer = torch.optim.AdamW(
        model_param_list + embed_param_list,
        lr=lr,
    )

    if pad_token_id is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    transformer.train()
    token_embedding.train()
    positional_embedding.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            # (B, T)
            tokens = batch.to(device)

            # input_tokens:  (B, T-1)
            # target_tokens: (B, T-1)
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            B, T = input_tokens.shape

            # (T-1, d_model)
            pos_enc = sinusoidal_positional_encoding(T, d_model, device)

            # (B, T-1, d_model)
            x_emb3 = tok_to_emb(input_tokens, emb3) + pos_enc.unsqueeze(0)
            y_emb3 = tok_to_emb(target_tokens, emb3) + pos_enc.unsqueeze(0)

            x_emb4 = tok_to_emb(input_tokens, emb4) + pos_enc.unsqueeze(0)
            y_emb4 = tok_to_emb(target_tokens, emb4) + pos_enc.unsqueeze(0)

            x_emb5 = tok_to_emb(input_tokens, emb5) + pos_enc.unsqueeze(0)
            y_emb5 = tok_to_emb(target_tokens, emb5) + pos_enc.unsqueeze(0)

            x_emb6 = tok_to_emb(input_tokens, emb6) + pos_enc.unsqueeze(0)
            y_emb6 = tok_to_emb(target_tokens, emb6) + pos_enc.unsqueeze(0)

            # causal_mask: (T, T)
            causal_mask = make_causal_mask(T, device)

            # (B, T, vocab_size)
            logits3 = transformer(x_emb3, y_emb3, causal_mask)
            logits4 = transformer(x_emb4, y_emb4, causal_mask)
            logits5 = transformer(x_emb5, y_emb5, causal_mask)
            logits6 = transformer(x_emb6, y_emb6, causal_mask)

            loss = criterion(
                logits.reshape(B * T, vocab_size),
                target_tokens.reshape(B * T),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            num_tokens = B * T
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        print(f"epoch {epoch + 1}: loss={avg_loss:.4f}, ppl={ppl:.2f}")
