import random
import math
import numpy as np
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


class PreTrainRecord:
    def __init__(
            self,
            job_id: string,
            num_epochs: int,
            embeddings,
            mask_token_id: int,
            pad_token_id: int,
            k_choices: list):

        self.job_id = job_id

        self.num_epochs = num_epochs
        self.embeddings = embeddings
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        # avg train loss per epoch
        self.train_loss_record = np.zeros(num_epochs)
        # ppl per epoch
        self.train_ppl_record = np.zeros(num_epochs)
        # tokens used per epoch per k-mer
        self.train_tokens_record = {k: np.zeros(num_epochs) for k in k_choices}

        # avg val loss per epoch
        self.val_loss_record = np.zeros(num_epochs)
        # avg val loss per epoch
        self.val_ppl_record = np.zeros(num_epochs)
        # tokens used per epoch per k-mer
        self.train_val_record = {k: np.zeros(num_epochs) for k in k_choices}

    def log_train_loss(
            self,
            epoch,
            loss,
            ppl):
        pass

    def log_train_tokens(
            self,
            epoch,
            k,
            num_tokens):

        self.train_tokens_record[k][epoch] += num_tokens

    def log_val_loss(
            self,
            epoch,
            avg_loss,
            ppl):
        pass

    def log_val_tokens(
            self,
            epoch,
            k,
            num_tokens):

        self.val_tokens_record[k][epoch] += num_tokens

    def visualize_records_to_png(self):
        pass

    def save_records_to_json(self):
        pass


def checkpoint_model_weights():
    pass


def checkpoint_embeddings():
    pass


def train_epoch():
    pass


def val_epoch():
    pass


def train(
        recorder: PreTrainRecord,
        model_config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        lr: float = 3e-4,
        mask_token_id: int = 0,
        pad_token_id: int | None = None,
        k_choices: list | None = None,
        emb_dict: dict | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """ Trains k-mer embeddings on their respective next k-mer token 
        prediction tasks via vanilla transformer models.

    Args:
        model_config (dict): _description_
        train_loader (DataLoader): _description_
        val_loader (DataLoader): _description_
        num_epochs (int): _description_
        lr (float, optional): _description_. Defaults to 3e-4.
        pad_token_id (int | None, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cuda"iftorch.cuda.is_available()else"cpu".
    """

    device = torch.device(device)

    d_model = model_config['d_model']
    num_heads = model_config['num_heads']
    dff = model_config['dff']
    num_enc_layers = model_config['num_enc_layers']
    num_dec_layers = model_config['num_dec_layers']

    if emb_dict is None:
        emb_dict = {
            3: nn.Embedding(64, d_model),
            4: nn.Embedding(256, d_model),
            5: nn.Embedding(1024, d_model),
            6: nn.Embedding(4096, d_model)
        }

    if k_choices is None:
        k_choices = emb_dict.keys()

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

            k = random.choice(k_choices)
            vocab_size = (4 ** k)
            emb = emb_dict[k]

            masked_tokens, patch_mask = mask_tokens(tokens, mask_token_id)

            input_emb = tok_to_emb(masked_tokens, emb)
            output_emb = tok_to_emb(tokens, emb)

            # (B, T, vocab_size)
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
