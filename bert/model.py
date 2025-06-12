from torch import nn
import torch
from torch.nn import functional as F
import math
from d2l import torch as d2l


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    # q,k,v = (B, N, d)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        d = q.size(dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = F.softmax(scores, dim=-1) # (B,N,N)
        return torch.bmm(self.attention_weights, v) # (B, N, d)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_embd, num_heads, dropout=0.0, device=None):
        super().__init__()
        self.W_q = nn.Linear(d_embd, d_embd, bias=False, device=device)
        self.W_k = nn.Linear(d_embd, d_embd, bias=False, device=device)
        self.W_v = nn.Linear(d_embd, d_embd, bias=False, device=device)
        self.W_o = nn.Linear(d_embd, d_embd, bias=False, device=device)
        self.num_heads = num_heads
        self.d_embd = d_embd
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.attention = DotProductAttention()

    # q,k,v = (B,n,d)
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        attention = self.attention(q, k, v) # (B*heads, n, d/heads)
        attention = attention.reshape((-1, self.num_heads, attention.shape[1], attention.shape[2]))
        attention = attention.transpose(1, 2) # (B, n, heads, d/heads)
        attention = attention.reshape((attention.shape[0], attention.shape[1], -1)) # (B,n,d)
        output = self.W_o(attention)
        return self.dropout(output) # (B, n, d)

    # x = (B,n,d)
    def split_heads(self, x:torch.Tensor)->torch.Tensor:
        x = x.reshape((x.shape[0], x.shape[1], self.num_heads, -1)) # (B, n, heads, d/heads)
        x = x.transpose(1, 2) # (B, heads, n, d/heads)
        return x.reshape((-1, x.shape[2], x.shape[3]))
    

class EncoderBlock(nn.Module):
    def __init__(self, d_embd, num_heads, ffn_hidden, dropout=0.0, device=None):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            d_embd=d_embd,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_embd, ffn_hidden, device=device),
            nn.Linear(ffn_hidden, d_embd, device=device),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_embd)
        self.ln2 = nn.LayerNorm(d_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X:torch.Tensor):
        attention = self.attention(self.ln1(X))
        out = self.ffn(self.ln2(attention))
        return self.dropout(out)
    
class BERTConfig:
    def __init__(self, d_embd, num_heads, ffn_hidden, vocab_size, num_layers=12, dropout=0.0, device=None):
        self.d_embd = d_embd
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.ffn_hidden = ffn_hidden
        self.dropout = dropout
        self.device = device
        self.num_layers = num_layers


class BERT(nn.Module):
    def __init__(self, conf: BERTConfig):
        super().__init__()
        self.model = nn.ModuleDict(dict(
            word_embd = nn.Embedding(conf.vocab_size, conf.d_embd),
            layers = nn.ModuleList([
                EncoderBlock(conf.d_embd, conf.num_heads, conf.ffn_hidden, conf.dropout, conf.device)
                for i in range(conf.num_layers)
            ]),
            ln = nn.LayerNorm(conf.d_embd),
            linear = nn.Linear(conf.d_embd, conf.vocab_size),
        ))

    def predict():
        pass
