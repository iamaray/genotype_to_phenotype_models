import torch
import torch.nn as nn

from g2p_gpt.model import G2P_GPT

def main():
    model = G2P_GPT(4, 16, 128)
    ex = torch.Tensor([[1, 2, 3, 4]])
    out = model(ex)
    
    print(out['3mer'].shape)
    print(out['4mer'].shape)
    print(out['5mer'].shape)
    print(out['6mer'].shape)
    
    
if __name__ == "__main__":
    main()