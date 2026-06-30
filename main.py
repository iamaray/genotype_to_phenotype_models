from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from helpers import make_causal_mask, tok_to_emb
from g2p_gpt.model import Transformer
from g2p_gpt.trainer import train


K_CHOICES = [3, 4, 5, 6]


class DummyKmerDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int):
        self.tokens = {
            k: torch.randint(0, 4 ** k, (num_samples, seq_len))
            for k in K_CHOICES
        }

    def __len__(self):
        return next(iter(self.tokens.values())).size(0)

    def __getitem__(self, idx):
        return {
            k: tokens[idx]
            for k, tokens in self.tokens.items()
        }


def make_dummy_loader(
        num_batches: int = 3,
        batch_size: int = 64,
        seq_len: int = 100):
    num_samples = num_batches * batch_size
    return DataLoader(
        DummyKmerDataset(num_samples, seq_len),
        batch_size=batch_size,
        shuffle=False,
    )


def make_model_config(model_id: str, causal: bool):
    return {
        "id": model_id,
        "causal": causal,
        "d_model": 16,
        "num_heads": 2,
        "dff": 32,
        "num_enc_layers": 6,
        "num_dec_layers": 6,
    }


def make_embedding_dict(d_model: int, device: torch.device):
    return {
        k: nn.Embedding(4 ** k, d_model).to(device)
        for k in K_CHOICES
    }


def load_pretrained_components(
        job_name: str,
        model_config: dict,
        device: torch.device):
    checkpoint_dir = Path("pretraining") / job_name
    model = Transformer(
        model_config["id"],
        model_config["num_heads"],
        model_config["d_model"],
        model_config["dff"],
        model_config["num_enc_layers"],
        model_config["num_dec_layers"],
        k_choices=K_CHOICES,
        causal=model_config["causal"],
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_dir / "model_weights.pt", map_location=device)
    )

    emb_dict = make_embedding_dict(model_config["d_model"], device)
    embedding_state = torch.load(
        checkpoint_dir / "embeddings.pt",
        map_location=device,
    )
    for k, emb in emb_dict.items():
        emb.load_state_dict(embedding_state[k])

    model.eval()
    for emb in emb_dict.values():
        emb.eval()
    return model, emb_dict


def test_loaded_components(
        model: Transformer,
        emb_dict: dict,
        causal: bool,
        device: torch.device,
        batch_size: int = 2,
        seq_len: int = 8):
    with torch.no_grad():
        for k in K_CHOICES:
            tokens = torch.randint(0, 4 ** k, (batch_size, seq_len), device=device)
            if causal:
                tokens = tokens[:, :-1]

            embeddings = tok_to_emb(tokens, emb_dict[k])
            mask = (
                make_causal_mask(tokens.size(1), device=device)
                if causal
                else None
            )
            logits = model(
                embeddings,
                embeddings,
                k,
                src_mask=mask,
                tgt_mask=mask,
                memory_mask=mask,
            )
            assert logits.shape == (batch_size, tokens.size(1), 4 ** k)


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")

    train_loader = make_dummy_loader()
    val_loader = make_dummy_loader()

    for causal in (False, True):
        job_name = "dummy_causal" if causal else "dummy_acausal"
        model_config = make_model_config(job_name, causal)
        train(
            job_name=job_name,
            model_config=model_config,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            k_choices=K_CHOICES,
            checkpoint_frequency=1000,
            device=str(device),
        )
        model, emb_dict = load_pretrained_components(
            job_name,
            model_config,
            device,
        )
        test_loaded_components(model, emb_dict, causal, device)
        print(f"loaded checkpoint smoke test passed for {job_name}")


if __name__ == "__main__":
    main()
