from torch import nn
import torch
import math
from corpus.vocab import Vocab

class DotProductAttention(nn.Module):
    def __init__(self, d_embd, dropout=0.0, device=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.tril(torch.ones(d_embd, d_embd, device=device)).view(1, d_embd, d_embd) # 下三角矩阵，对角线上方0下方1

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        d = q.shape[-1]
        n = q.shape[-2]
        # q: (B, n, d), k: (B, n, d), v: (B, n, d)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d) # (B, n, n)
        scores = scores.masked_fill(self.mask[:, :n, :n] == 0, float('-inf'))
        # (B, n, n) 表示n个词，每个词对其他词的注意力权重
        self.attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), v) # (B, nq, d)

# 多头自注意力, GPT只使用自注意力
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, d_embd, dropout=0.0, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_embd = d_embd
        self.device = device
        # 注意力对齐
        self.W_q = nn.Linear(d_embd, d_embd*num_heads, device=device, bias=False)
        self.W_k = nn.Linear(d_embd, d_embd*num_heads, device=device, bias=False)
        self.W_v = nn.Linear(d_embd, d_embd*num_heads, device=device, bias=False)
        # 输出
        self.W_out = nn.Linear(d_embd*num_heads, d_embd, device=device, bias=False)
        # 点积注意力
        self.attention = DotProductAttention(d_embd, dropout=dropout, device=device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X:torch.Tensor):
        # X:(B,n,d_embd)
        # 自注意力，qkv都是X
        q = self.W_q(X)
        k = self.W_k(X)
        v = self.W_v(X)
        # 拆分多头
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v) # (B*num_heads, n, d_embd/num_heads)
        # 注意力
        attention = self.attention(q, k, v) # (B*num_heads, n, d_embd/num_heads)
        attention = self.merge_attention(attention) # (B, n, d_embd)
        return self.dropout(self.W_out(attention))

    def split_heads(self, X: torch.Tensor)->torch.Tensor:
        # X:(B,n,d_embd)
        X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1)) # (B, n, num_heads, d_embd/num_heads)
        X = X.transpose(1, 2) # (B, num_heads, n, d_embd/num_heads)
        return X.reshape((-1, X.shape[2], X.shape[3])) # (B*num_heads, n, d_embd/num_heads)
    
    def merge_attention(self, X: torch.Tensor)->torch.Tensor:
        # X: (B*num_heads, n, d_embd/num_heads)
        X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2])) # (B, num_heads, n, d_embd/num_heads)
        X = X.transpose(1, 2) # (B, n, num_heads, d_embd/num_heads)
        return X.reshape((X.shape[0], X.shape[1], -1)) # (B, n, d_embd)
    
    @property
    def attention_weights(self)->torch.Tensor:
        return self.attention.attention_weights

class TransformerBlock(nn.Module):
    def __init__(self, d_embd, num_heads, d_ff, dropout=0.0, device=None):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_embd, d_ff, device=device),
            nn.Linear(d_ff, d_embd, device=device),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.attention = MultiHeadSelfAttention(
            num_heads=num_heads,
            d_embd=d_embd,
            dropout=dropout,
            device=device
        )
        self.ln1 = nn.LayerNorm(d_embd, device=device)
        self.ln2 = nn.LayerNorm(d_embd, device=device)

    def forward(self, X:torch.Tensor):
        # 前置LayerNorm
        Y = X + self.attention(self.ln1(X))
        Y = Y + self.ffn(self.ln2(Y))
        return Y

class GPTConfig:
    def __init__(self, vocab_size, block_size, d_embd, num_heads, d_ff, dropout, num_layers, device):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_embd = d_embd
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_layers = num_layers
        self.deivce = device


class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.device = config.deivce
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_embd, device=self.device),  # 词嵌入
            wpe = nn.Embedding(config.block_size, config.d_embd, device=self.device),  # 位置编码
            embd_drop = nn.Dropout(config.dropout),
            layers =  nn.ModuleList([TransformerBlock(config.d_embd, 
                                        config.num_heads, 
                                        config.d_ff, 
                                        config.dropout, 
                                        device=self.device) for _ in range(config.num_layers)]), # transformer layers
            ln = nn.LayerNorm(config.d_embd, device=self.device),
            out = nn.Linear(config.d_embd, config.vocab_size, device=self.device) # 输出映射
        ))
        self.block_size = config.block_size

    def init_params(self):
        pass
    
    def forward(self, X: torch.Tensor):
        n = X.shape[1]
        pos = torch.arange(0, n, 1, device=X.device, dtype=X.dtype).view(1, -1)
        wte = self.transformer.wte(X)
        wpe = self.transformer.wpe(pos)
        y = self.transformer.embd_drop(wte + wpe)
        for block in self.transformer.layers:
            y = block(y)
        y = self.transformer.ln(y)
        out = self.transformer.out(y)
        return out
    
    def attention_weights(self)->list[torch.Tensor]:
        return [block.attention.attention_weights for block in self.transformer.layers]
    
    def generate(self, input: torch.Tensor, max_tokens: int, end_token:float):
        self.eval()
        output = [token.item() for token in input[0]]
        with torch.no_grad():
            for _ in range(max_tokens):
                # 去除input超出block_size的部分
                if input.shape[1] > self.block_size:
                    input = input[:, -self.block_size:] 
                logits = self(input) # (1, n, d_embd)
                logits = logits[:, -1, :] # (1, d_embd) 只取最后一个token的logits
                probs = torch.nn.functional.softmax(logits, dim=-1) # (1, vocab_size)
                next_token = torch.multinomial(probs, num_samples=1) # (1, 1) 从probs中采样一个token
                input = torch.cat((input, next_token), dim=1) # (1, n+1) 把采样的token加到输入中
                output.append(next_token.item()) # 把采样的token加到输出中
                if next_token.item() == end_token: # 如果采样的token是结束符，则停止
                    break
        return output
