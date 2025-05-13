from .vocab import Vocab
import torch
import jieba
import re

def load_en_zh_data(batch_size=64, num_steps=10, num_examples=10000):
    with open("D:\\data\\training\\en-cn\\cmn.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [re.sub(r"[,.!?]", " ", line).split('\t') for line in lines[:num_examples]]
    # 中英文分词
    en_tokens = [line[0].strip().lower().split(' ') for line in lines]
    zh_tokens = [jieba.lcut(line[1].strip()) for line in lines]
    # 创建词库
    en_vocab = Vocab(en_tokens)
    zh_vocab = Vocab(zh_tokens)
    # 填充、截断、添加结束标记
    for token_list in en_tokens:
        if len(token_list) >= num_steps:
            token_list[num_steps-1] = '<eos>'
            continue
        token_list.append('<eos>')
        token_list.extend(['<pad>'] * (num_steps - len(token_list)))
    # 填充、截断、添加结束标记
    for token_list in zh_tokens:
        if len(token_list) >= num_steps:
            token_list[num_steps-1] = '<eos>'
            continue
        token_list.append('<eos>')
        token_list.extend(['<pad>'] * (num_steps - len(token_list)))
    
    data = []
    # 划分批次训练数据
    for i in range(0, len(en_tokens), batch_size):
        en_batch = en_tokens[i:i+batch_size]
        zh_batch = zh_tokens[i:i+batch_size]
        en_batch = [en_vocab[en[:num_steps]] for en in en_batch]
        zh_batch = [zh_vocab[zh[:num_steps]] for zh in zh_batch]
        en_valid_len = [len(en) for en in en_batch]
        zh_valid_len = [len(zh) for zh in zh_batch]
        data.append((
            torch.tensor(en_batch, dtype=torch.long), torch.tensor(en_valid_len, dtype=torch.long),
            torch.tensor(zh_batch, dtype=torch.long), torch.tensor(zh_valid_len, dtype=torch.long)
        ))
    return data, en_vocab, zh_vocab 

