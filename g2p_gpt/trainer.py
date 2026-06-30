import random
import math
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from helpers import make_causal_mask, tok_to_emb
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
            model_id: str,
            job_type: str,
            num_epochs: int,
            embeddings,
            mask_token_id: int,
            pad_token_id: int | None,
            k_choices: list):

        self.job_id = job_id
        self.model_id = model_id
        if job_type not in ("next-token", "patch"):
            raise ValueError("Unrecognized job type.")
        self.job_type = job_type

        self.num_epochs = num_epochs
        self.embeddings = embeddings
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.trained_causally = False
        self.trained_acausally = False

        self.train_loss_record = [0.0] * num_epochs
        self.train_ppl_record = [0.0] * num_epochs
        self.train_tokens_record = {k: [0] * num_epochs for k in k_choices}

        self.val_loss_record = [0.0] * num_epochs
        self.val_ppl_record = [0.0] * num_epochs
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

    def mark_trained(self, causal: bool):
        if causal:
            self.trained_causally = True
        else:
            self.trained_acausally = True

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
            "model_id": self.model_id,
            "job_type": self.job_type,
            "trained_causally": self.trained_causally,
            "trained_acausally": self.trained_acausally,
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
    torch.save(_state_dict_to_cpu(model.state_dict()), path)
    return path


def checkpoint_embeddings(emb_dict, path):
    torch.save({
        k: _state_dict_to_cpu(emb.state_dict())
        for k, emb in emb_dict.items()
    }, path)
    return path


def _state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def checkpoint_pretraining_state(model, emb_dict, recorder, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_model_weights(model, checkpoint_dir / "model_weights.pt")
    checkpoint_embeddings(emb_dict, checkpoint_dir / "embeddings.pt")
    recorder.save_records_to_json(checkpoint_dir / "pretrain_record.json")
    recorder.visualize_records_to_png(checkpoint_dir / "train_val_loss.png")
    return checkpoint_dir


def _batch_tokens(batch, k, device):
    if isinstance(batch, dict):
        tokens = batch[k]
    else:
        tokens = batch
    return tokens.to(device, non_blocking=True)


def _run_causal_batch(
        transformer,
        emb_dict,
        batch,
        criterion,
        recorder,
        k_choices,
        _mask_token_id,
        device):
    k = random.choice(k_choices)
    tokens = _batch_tokens(batch, k, device)
    input_tokens = tokens[:, :-1]
    targets = tokens[:, 1:]
    seq_len = input_tokens.size(1)
    if seq_len == 0:
        return None

    causal_mask = make_causal_mask(seq_len, device=device)
    key_padding_mask = (
        input_tokens.eq(recorder.pad_token_id)
        if recorder.pad_token_id is not None
        else None
    )
    logits = transformer(
        tok_to_emb(input_tokens, emb_dict[k]),
        tok_to_emb(input_tokens, emb_dict[k]),
        k,
        src_mask=causal_mask,
        tgt_mask=causal_mask,
        memory_mask=causal_mask,
        src_key_padding_mask=key_padding_mask,
        tgt_key_padding_mask=key_padding_mask,
    )
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    num_tokens = (
        targets.ne(recorder.pad_token_id).sum().item()
        if recorder.pad_token_id is not None
        else targets.numel()
    )
    return loss, num_tokens, k


def _run_acausal_batch(
        transformer,
        emb_dict,
        batch,
        criterion,
        _recorder,
        k_choices,
        mask_token_id,
        device):
    k = random.choice(k_choices)
    tokens = _batch_tokens(batch, k, device)
    masked_tokens, patch_mask = mask_tokens(tokens, mask_token_id)
    logits = transformer(
        tok_to_emb(masked_tokens, emb_dict[k]),
        tok_to_emb(tokens, emb_dict[k]),
        k,
    )
    targets = tokens[patch_mask]
    loss = criterion(logits[patch_mask], targets)
    return loss, targets.numel(), k


def _update_from_batch(loss, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _log_batch_tokens(recorder, is_train, epoch, k, num_tokens):
    if is_train:
        recorder.log_train_tokens(epoch, k, num_tokens)
    else:
        recorder.log_val_tokens(epoch, k, num_tokens)


def _embedding_device(embedding):
    return next(embedding.parameters()).device


def _validate_embedding_devices(emb_dict, device):
    mismatched = [
        k for k, emb in emb_dict.items()
        if _embedding_device(emb) != device
    ]
    if mismatched:
        raise ValueError(
            f"Embeddings for k={mismatched} are not on model device {device}."
        )


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
    device = transformer.device
    transformer.train(is_train)
    for emb in emb_dict.values():
        emb.train(is_train)
    _validate_embedding_devices(emb_dict, device)

    is_causal = transformer.causal
    batch_step = _run_causal_batch if is_causal else _run_acausal_batch
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            result = batch_step(
                transformer,
                emb_dict,
                batch,
                criterion,
                recorder,
                k_choices,
                mask_token_id,
                device,
            )
            if result is None:
                continue

            loss, num_tokens, k = result
            if is_train:
                _update_from_batch(loss, optimizer)

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            _log_batch_tokens(recorder, is_train, epoch, k, num_tokens)

    avg_loss = total_loss / total_tokens if total_tokens else float("nan")
    ppl = math.exp(avg_loss)
    if is_train:
        transformer.trained_causally = (
            transformer.trained_causally or is_causal
        )
        transformer.trained_acausally = (
            transformer.trained_acausally or not is_causal
        )
        recorder.mark_trained(is_causal)
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
        job_name: str,
        model_config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        lr: float = 3e-4,
        mask_token_id: int = 0,
        pad_token_id: int | None = None,
        k_choices: list | None = None,
        emb_dict: dict | None = None,
        checkpoint_frequency: int = 1000,
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
    assert checkpoint_frequency >= 1

    model_id = model_config.get('id', job_name)
    is_causal = model_config.get('causal', False)
    job_type = model_config.get(
        'job_type',
        "next-token" if is_causal else "patch",
    )
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

    transformer = Transformer(
        model_id,
        num_heads,
        d_model,
        dff,
        num_enc_layers,
        num_dec_layers,
        k_choices=k_choices,
        causal=is_causal,
    ).to(device)
    emb_dict = {k: emb.to(device) for k, emb in emb_dict.items()}
    recorder = PreTrainRecord(
        job_id=job_name,
        model_id=model_id,
        job_type=job_type,
        num_epochs=num_epochs,
        embeddings=emb_dict,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        k_choices=k_choices,
    )
    embed_param_list = [
        param for emb in emb_dict.values() for param in emb.parameters()
    ]
    optimizer = torch.optim.AdamW(
        list(transformer.parameters()) + embed_param_list,
        lr=lr,
    )

    if pad_token_id is None:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)

    checkpoint_dir = Path("pretraining") / job_name
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

        checkpoint_num = epoch + 1
        if checkpoint_num % checkpoint_frequency == 0:
            checkpoint_pretraining_state(
                transformer,
                emb_dict,
                recorder,
                checkpoint_dir / str(checkpoint_num),
            )

    checkpoint_pretraining_state(
        transformer, emb_dict, recorder, checkpoint_dir)
    return transformer, emb_dict, recorder
