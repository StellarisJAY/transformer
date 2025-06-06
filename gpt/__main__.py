from .model import GPT, GPTConfig
import torch

if __name__ == "__main__":
    batch_size = 16
    conf = GPTConfig(
        vocab_size=2048,
        block_size=8,
        d_embd=512,
        num_heads=8,
        d_ff=1024,
        dropout=0.0,
        num_layers=12,
        device=torch.device('cuda0') if torch.cuda.is_available() else torch.device('cpu')
    )
    gpt = GPT(conf)
    X = torch.ones((batch_size, conf.block_size), dtype=torch.long, device=conf.deivce)
    Y = gpt(X)
    print(Y.shape)

    

