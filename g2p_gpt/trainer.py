import random
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..helpers import make_causal_mask, sinusoidal_positional_encoding, tok_to_emb
from .model import Transformer


def mask_tokens(tokens: torch.Tensor, mask_token_id: int, device: torch.device):
    # tokens : (B, T)
    # returns patched tokens (B,T) and a boolean patch mask (True where tokens patched)

    masked_tokens = tokens.clone()
    patch_mask = torch.zeros(
        tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=device)

    for b in range(tokens.shape[0]):
        a = random.randint(0, T - 1)
        b = random.randint(0, T - 1)
        while math.fabs(a - b) <= 2:
            b = random.randint(0, T - 1)

        start = min(a, b)
        end = max(a, b)

        masked_tokens[b, start:end] = mask_token_id
        patch_mask[b, start:end] = True

    return masked_tokens, patch_mask


def train(
        params: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        lr: float = 3e-4,
        mask_token_id: int = 0,
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

    emb_dict = {
        3: nn.Embedding(64, d_model),
        4: nn.Embedding(256, d_model),
        5: nn.Embedding(1024, d_model),
        6: nn.Embedding(4096, d_model)
    }

    transformer = Transformer(num_heads, d_model, dff,
                              num_enc_layers, num_dec_layers)
    embed_param_list = list(model.parameters()) + list(emb3.parameters()) + list(
        emb4.parameters()) + list(emb5.parameters()) + list(emb6.parameters())
    optimizer = torch.optim.AdamW(
        list(transformer.parameters()) + embed_param_list,
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
            B, T = input_tokens.shape

            k = random.choice([3, 4, 5, 6])
            emb = emb_dict[k]

            masked_tokens, patch_mask = mask_tokens(tokens, 0)

            input_emb = tok_to_emb(masked_tokens, emb)
            output_emb = tok_to_emb(tokens, emb)

            logits = transformer(input_emb, output_emb, k)

            loss = criterion(
                logits[patch_mask].reshape(B * T, vocab_size),
                tokens[patch_mask].reshape(B * T),
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
