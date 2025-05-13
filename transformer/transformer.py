from torch import nn
import torch
from d2l import torch as d2l
import math
from torch.nn import functional as F
from datetime import datetime
import builtins

# 掩蔽softmax
# 输入 X.shape = (B, n, d)
# 输入 valid_lens.shape = (B) 或 (B, n)
def masked_softmax(X, valid_lens:None):
    # 没有长度限制，直接在最后一个维度softmax
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    shape = X.shape
    # 把valid_lens展开，获得每个批次中每个d的长度限制
    # valid_lens.shape = (B * n)
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    # 降维与valid_lens对齐
    # X.shape = (B * n, d)
    X = torch.reshape(X, (-1, shape[-1]))

    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32)[:] < valid_lens[:, None]
    X[~mask] = -1e6
    return nn.functional.softmax(torch.reshape(X, shape), dim=-1)

# 点积注意力
class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    # 因为多头注意力并行计算，此时的B是初始的批次B0*num_heads
    # 输入 Q.shape = (B, nq, d)
    # 输入 K.shape = (B, nkv, d)
    # 输入 V.shape = (B, nkv, size_v)
    # 输入 valid.shape = (B) 或 (B, nq)
    # 输出 attention.shape = (B, nq, size_v)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # scores.shape = (B, nq, nkv)
        scores = torch.bmm(queries, torch.transpose(keys, 1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 多头注意力
class MultiheadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout=0):
        super(MultiheadAttention, self).__init__()
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=False)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)
    
    # 输入 q.shape = (B, nq, size_q)
    # 输入 k.shape = (B, nkv, size_k)
    # 输入 v.shape = (B, nkv, size_v)
    # 输入 valid.shape = (B) 或 (B, nq)
    def forward(self, queries, keys, values, valid_lens):
        # 经过全连接层后，qk维度对齐
        q = self.W_q(queries)
        k = self.W_k(keys)
        v = self.W_v(values)
        # 拆分qkv，使多个注意力头并行计算
        # qk.shape = (B*num_heads, n, num_hiddens/num_heads)
        # v.shape = (B*num_heads, n, num_hiddens/num_heads)
        q, k, v = self.split_qkv(q), self.split_qkv(k), self.split_qkv(v)
        if valid_lens is not None:
            # valid_lens的初始shape=(B)，需要与qkv对齐(B*num_heads)
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # 多个点积注意力头输出 attentions.shape = (B*num_heads, qn, num_hiddens/num_heads)
        attentions = self.attention(q, k, v, valid_lens)
        # 多头注意力汇聚 output.shape = (B, nq, num_hiddens)
        output = self.W_o(self.merge_attentions(attentions))
        return output

    # 多头注意力开始的qkv全连接层是并行计算的，得到的结果需要拆分成每个头单独一个qkv
    # 并且要保持三维，使点积注意力也能并行计算
    # 输入 X.shape = (B, n, num_hiddens)
    # 输出 X.shape = (B*num_heads, n, num_hiddens/num_heads)
    def split_qkv(self, X):
        X = torch.reshape(X, (X.shape[0], X.shape[1], self.num_heads, -1))
        X = torch.transpose(X, 1, 2)
        return torch.reshape(X, (-1, X.shape[2], X.shape[3]))

    # 将分散的多头注意力矩阵汇聚
    # 输入 X.shape = (B * num_heads, nq, num_hiddens/num_heads)
    # 输出 X.shape  = (B, nq, num_hiddens)
    def merge_attentions(self, X):
        X = torch.reshape(X, (-1, self.num_heads, X.shape[1], X.shape[2]))
        X = torch.transpose(X, 1, 2)
        return torch.reshape(X, (X.shape[0], X.shape[1], -1))

# 位置编码
class PositionEncoding(nn.Module):
    def __init__(self, d, max_len=1000):
        super(PositionEncoding, self).__init__()
        self.P = torch.zeros((1, max_len, d))
        pos = torch.arange(max_len, dtype=torch.float).reshape(-1, 1)
        freq = torch.pow(10000, torch.arange(0, d, 2, dtype=torch.float) / d)
        X = pos / freq
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    # 输入 X.shape = (B, n, d)
    # 输出 (X+P).shape = (B, n, d)
    def forward(self, X):
        return X + self.P[:, :X.shape[1], :].to(X.device)
# MLP
class PointwiseFFN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(PointwiseFFN, self).__init__()
        self.dense1 = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.LeakyReLU()
        self.dense2 = nn.Linear(num_hiddens, num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# 残差+层规范化
class AddNorm(nn.Module):
    def __init__(self, normalize_shape, dropout=0):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape=normalize_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# 编码器块
class EncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, 
                 num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, dropout=0):
        super(EncoderBlock, self).__init__()
        # 多头注意力
        self.attention = MultiheadAttention(query_size, key_size, value_size, 
                                            num_hiddens, num_heads, dropout=dropout)
        # 残差+规范化
        self.add_norm1 = AddNorm(norm_shape, dropout)
        # MLP
        self.ffn = PointwiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        # 残差+规范化
        self.add_norm2 = AddNorm(norm_shape, dropout)
    
    # 输入 X.shape = (B, n, d)，位置编码后的嵌入层输出
    # 输出 Y.shape = (B, n, d)，自注意力输入输出大小相同
    def forward(self, X, valid_lens):
        # 自注意力，QKV都是X
        attention = self.attention(X, X, X, valid_lens)
        # 自注意力之后的残差+规范化
        X = self.add_norm1(X, attention)
        # 多层感知机
        Y = self.ffn(X)
        # MLP之后的残差+规范化
        return self.add_norm2(X, Y)

# Transformer 编码器
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, value_size, num_hiddens, 
                 num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, 
                 num_layers, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 位置编码
        self.position_encoding = PositionEncoding(num_hiddens)
        # N个编码器块
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f'enc_blk_{i}', 
                                   EncoderBlock(query_size, key_size, value_size, 
                                                num_hiddens, num_heads, norm_shape, 
                                                ffn_num_inputs, ffn_num_hiddens, dropout=dropout))

    # 输入X.shape = (B,n) 长度为n的词元序列
    # 输入valid_lens.shape = (B)
    def forward(self, X, valid_lens):
        # 嵌入层
        embedding = self.embedding(X)
        # 位置编码
        X = self.position_encoding(embedding * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blocks)
        # n个编码器块
        for i, blk in enumerate(self.blocks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

# 解码器块
class DecoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, 
                 num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, i, dropout=0):
        super(DecoderBlock, self).__init__()
        # 解码器块的序号
        self.i = i
        # 掩蔽多头注意力（解码器自注意力）
        self.attention1 = MultiheadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout=dropout)
        # 掩蔽多头注意力之后的残差+规范化
        self.add_norm1 = AddNorm(norm_shape, dropout)
        # 编码器KV解码器Q的多头注意力（编码器+解码器注意力）
        self.attention2 = MultiheadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout=dropout)
        # 残差+规范化
        self.add_norm2 = AddNorm(norm_shape, dropout)
        # MLP
        self.ffn = PointwiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        # 残差+规范化
        self.add_norm3 = AddNorm(norm_shape, dropout)

    # 输入 X.shape = (B, n, d)
    # 输入 state，编码器状态，state = (编码器输出，编码器有效长度，解码器直到当前时间步每个块的输出)
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            # 训练阶段，X是完整的序列
            key_value = X
        else:
            # 预测阶段
            key_value = torch.cat((state[2][self.i], X), dim=1)
            # 自回归，把当前时间步的state保存
            state[2][self.i] = key_value
        if self.training:
            batch_size, num_steps = X.shape[0], X.shape[1]
            # 设置每一个时间步的长度限制 
            dec_valid_lens = torch.arange(1, num_steps + 1, 1, dtype=torch.float).repeat(batch_size, 1)
        else:
            # 预测时不需要掩蔽，因为输入的X不是完整序列
            dec_valid_lens = None
        # 解码器自注意力（训练时掩蔽）
        Y1 = self.attention1(X, key_value, key_value, dec_valid_lens)
        X2 = self.add_norm1(X, Y1)
        # 编码器+解码器注意力
        Y2 = self.attention2(X2, enc_outputs, enc_outputs, enc_valid_lens)
        X3 = self.add_norm2(X2, Y2)
        # 全连接层
        Y3 = self.ffn(X3)
        Y = self.add_norm3(X3, Y3)
        return Y, state

# Transformer 解码器
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, value_size, num_hiddens, 
                 num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_layers, dropout=0):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 位置编码
        self.position_encoding = PositionEncoding(num_hiddens)
        # n个解码器块
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f'dec_blk_{i}', DecoderBlock(query_size, key_size, value_size, 
                                                                num_hiddens, num_heads, norm_shape, 
                                                                ffn_num_inputs, ffn_num_hiddens, i, dropout=dropout))
        # 全连接层，隐藏状态到输出的映射
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens):
        return (enc_outputs, enc_valid_lens, [None] * self.num_layers)

    # 输入X.shape = (B, n)
    def forward(self, X, state):
        # 嵌入
        embedding = self.embedding(X)
        # 位置编码
        pe = self.position_encoding(embedding * math.sqrt(self.num_hiddens))
        self.attention_weights = [[None] * len(self.blocks) for i in range(2)]
        blk_X, blk_state = pe, state
        # n个解码器块
        for i, block in enumerate(self.blocks):
            blk_X, blk_state = block(blk_X, blk_state)
            self.attention_weights[0][i] = block.attention1.attention.attention_weights
            self.attention_weights[1][i] = block.attention2.attention.attention_weights
        return self.dense(blk_X), blk_state
    
class Transformer(nn.Module):
    def __init__(self, encoder:TransformerEncoder, decoder:TransformerDecoder, num_steps):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_steps = num_steps
    
    def forward(self, enc_x, dec_x, enc_valid_lens):
        enc_outputs = self.encoder(enc_x, enc_valid_lens)
        state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(dec_x, state)


def train_transformer(model: Transformer, train_iter, tgt_vocab, lr=0.01, num_epochs=100):
    # Adam 梯度下降
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = d2l.MaskedSoftmaxCELoss()
    # 初始化参数
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    model.apply(xavier_init_weights)
    model.train()
    for epoch in range(num_epochs):
        ep_train_losses = []
        ep_start = datetime.now()
        for batch in train_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0]).reshape(-1, 1)
            # 解码器的输入是 <bos>开头然后去掉最后一个词元
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, state = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len).sum()
            l.backward()
            optimizer.step()
            ep_train_losses.append(l)
        with torch.no_grad():
            ep_train_loss = torch.tensor(ep_train_losses).mean()
            print(f'epoch:{epoch},train_loss={ep_train_loss},duration:{(datetime.now() - ep_start).total_seconds()}s')
    torch.save(model.state_dict(), 'transformer.params')

def load_params(model: Transformer, path:str='transformer.params'):
      # 加载上次训练的参数
    with torch.serialization.safe_globals([builtins.getattr]):
        try:
            model.load_state_dict(torch.load(path))
        except Exception as e:
            print(e)

def pad_sequence(X:torch.Tensor, pad_value, padded_size):
    pad_size = padded_size - X.size(-1)
    return F.pad(X, (0, pad_size), value=pad_value)

def transformer_predict(model: Transformer, sentence:str, src_vocab, tgt_vocab, delim=' '):
    model.eval()
    sequence = sentence.lower().split(' ')
    sequence.append('<eos>')
    valid_len = torch.tensor([len(sequence)])
    # x.shape = (1, valid_len)
    X = torch.tensor(src_vocab[sequence]).reshape(1,-1)
    # 填充序列，与transformer输入长度对齐
    X = pad_sequence(X, torch.tensor(src_vocab['<pad>']), model.num_steps)
    # 编码器输出
    enc_output = model.encoder(X, valid_len)
    dec_state = model.decoder.init_state(enc_output, valid_len)
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long),
        dim=0)
    output_seq = []
    for i in range(model.num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        dec_X = torch.argmax(Y, dim=2, keepdim=False).int()
        pred = tgt_vocab.to_tokens(dec_X)
        output_seq.append(pred)
        if pred == '<eos>':
            break
    return delim.join(output_seq)

def create_transformer(src_vocab_size, tgt_vocab_size, num_steps,
                       q_size=32, k_size=32, v_size=32, 
                       num_hiddens=32, num_heads=4, norm_shape=[32], 
                       ffn_num_inputs=32, ffn_num_hiddens=64, 
                       enc_layers=2, dec_layers=2, 
                       dropout=0.1):
    encoder = TransformerEncoder(src_vocab_size, q_size, k_size, v_size, num_hiddens, num_heads, 
                                 norm_shape, ffn_num_inputs, ffn_num_hiddens, enc_layers, dropout=dropout)
    decoder = TransformerDecoder(tgt_vocab_size, q_size, k_size, v_size, num_hiddens, num_heads, 
                                 norm_shape, ffn_num_inputs, ffn_num_hiddens, dec_layers, dropout=dropout)
    transformer = Transformer(encoder, decoder, num_steps)
    return encoder, decoder, transformer 

torch.serialization.add_safe_globals([Transformer, TransformerDecoder, TransformerEncoder, 
                                      DecoderBlock, EncoderBlock, PointwiseFFN, DotProductAttention, 
                                      MultiheadAttention, nn.Embedding, PositionEncoding, nn.Sequential, 
                                      nn.Linear, nn.ReLU, nn.Dropout, AddNorm, nn.LayerNorm, nn.LeakyReLU]) 