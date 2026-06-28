def make_causal_mask(T: int, device=None):
    # returns: (T, T)
    return torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device),
        diagonal=1,
    )


def sinusoidal_positional_encoding(
    max_len: int,
    d_model: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    position = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(1)
    # shape: (max_len, 1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / d_model)
    )
    # shape: (ceil(d_model / 2),)

    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)

    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

