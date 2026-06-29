import random
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from helpers import tok_to_emb
from .model import Transformer


def mask_tokens(
        tokens: torch.Tensor,
        mask_token_id: int,
        device: torch.device | None = None):
    # tokens : (B, T)
    # returns patched tokens (B,T) and a boolean patch mask (True where tokens patched)

    device = tokens.device if device is None else device
    masked_tokens = tokens.clone()
    B, T = tokens.shape
    patch_mask = torch.zeros(
        B, T, dtype=torch.bool, device=device)

    for i in range(B):
        a = random.randint(0, T - 1)
        b = random.randint(0, T - 1)
        while T > 3 and abs(a - b) <= 2:
            b = random.randint(0, T - 1)

        start = min(a, b)
        end = max(a, b) + 1

        masked_tokens[i, start:end] = mask_token_id
        patch_mask[i, start:end] = True

    return masked_tokens, patch_mask


class PreTrainRecord:
    def __init__(
            self,
            job_id: str,
            num_epochs: int,
            embeddings,
            mask_token_id: int,
            pad_token_id: int | None,
            k_choices: list):

        self.job_id = job_id

        self.num_epochs = num_epochs
        self.embeddings = embeddings
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        # avg train loss per epoch
        self.train_loss_record = [0.0] * num_epochs
        # ppl per epoch
        self.train_ppl_record = [0.0] * num_epochs
        # tokens used per epoch per k-mer
        self.train_tokens_record = {k: [0] * num_epochs for k in k_choices}

        # avg val loss per epoch
        self.val_loss_record = [0.0] * num_epochs
        # avg val loss per epoch
        self.val_ppl_record = [0.0] * num_epochs
        # tokens used per epoch per k-mer
        self.val_tokens_record = {k: [0] * num_epochs for k in k_choices}

    def log_train_loss(
            self,
            epoch,
            loss,
            ppl):
        self.train_loss_record[epoch] = loss
        self.train_ppl_record[epoch] = ppl

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
        self.val_loss_record[epoch] = avg_loss
        self.val_ppl_record[epoch] = ppl

    def log_val_tokens(
            self,
            epoch,
            k,
            num_tokens):

        self.val_tokens_record[k][epoch] += num_tokens

    def visualize_records_to_png(self, path):
        import matplotlib.pyplot as plt

        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, self.train_loss_record, label="train_loss")
        plt.plot(epochs, self.val_loss_record, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def save_records_to_json(self, path):
        records = {
            "job_id": self.job_id,
            "train_loss": self.train_loss_record,
            "train_ppl": self.train_ppl_record,
            "train_tokens": self.train_tokens_record,
            "val_loss": self.val_loss_record,
            "val_ppl": self.val_ppl_record,
            "val_tokens": self.val_tokens_record,
        }
        with open(path, "w") as f:
            json.dump(records, f)
        return path


def checkpoint_model_weights(model, path):
    torch.save(model.state_dict(), path)
    return path


def checkpoint_embeddings(emb_dict, path):
    torch.save({k: emb.state_dict() for k, emb in emb_dict.items()}, path)
    return path


def _batch_tokens(batch, device):
    tokens = batch[0] if isinstance(batch, (list, tuple)) else batch
    return tokens.to(device)


def _run_epoch(
        transformer,
        emb_dict,
        loader,
        criterion,
        recorder,
        epoch,
        k_choices,
        mask_token_id,
        device,
        optimizer=None):
    is_train = optimizer is not None
    transformer.train(is_train)
    for emb in emb_dict.values():
        emb.train(is_train)

    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            tokens = _batch_tokens(batch, device)
            k = random.choice(k_choices)
            masked_tokens, patch_mask = mask_tokens(tokens, mask_token_id)
            logits = transformer(
                tok_to_emb(masked_tokens, emb_dict[k]),
                tok_to_emb(tokens, emb_dict[k]),
                k,
            )
            targets = tokens[patch_mask]
            loss = criterion(logits[patch_mask], targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            num_tokens = targets.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            if is_train:
                recorder.log_train_tokens(epoch, k, num_tokens)
            else:
                recorder.log_val_tokens(epoch, k, num_tokens)

    avg_loss = total_loss / total_tokens if total_tokens else float("nan")
    ppl = math.exp(avg_loss)
    if is_train:
        recorder.log_train_loss(epoch, avg_loss, ppl)
    else:
        recorder.log_val_loss(epoch, avg_loss, ppl)
    return avg_loss, ppl


def train_epoch(
        transformer,
        emb_dict,
        train_loader,
        criterion,
        optimizer,
        recorder,
        epoch,
        k_choices,
        mask_token_id,
        device):
    return _run_epoch(
        transformer, emb_dict, train_loader, criterion, recorder,
        epoch, k_choices, mask_token_id, device, optimizer)


def val_epoch(
        transformer,
        emb_dict,
        val_loader,
        criterion,
        recorder,
        epoch,
        k_choices,
        mask_token_id,
        device):
    return _run_epoch(
        transformer, emb_dict, val_loader, criterion, recorder,
        epoch, k_choices, mask_token_id, device)


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
        k_choices = list(emb_dict.keys())

    transformer = Transformer(num_heads, d_model, dff,
                              num_enc_layers, num_dec_layers).to(device)
    emb_dict = {k: emb.to(device) for k, emb in emb_dict.items()}
    embed_param_list = [
        param for emb in emb_dict.values() for param in emb.parameters()
    ]
    optimizer = torch.optim.AdamW(
        list(transformer.parameters()) + embed_param_list,
        lr=lr,
    )

    if pad_token_id is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    for epoch in range(num_epochs):
        train_loss, train_ppl = train_epoch(
            transformer, emb_dict, train_loader, criterion, optimizer,
            recorder, epoch, k_choices, mask_token_id, device)
        val_loss, val_ppl = val_epoch(
            transformer, emb_dict, val_loader, criterion, recorder,
            epoch, k_choices, mask_token_id, device)

        print(
            f"epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_ppl={train_ppl:.2f}, "
            f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}"
        )

    return transformer, emb_dict, recorder
