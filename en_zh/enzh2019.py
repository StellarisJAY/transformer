import jieba
from ..corpus.vocab import Vocab
import torch

def load_train_data(num_lines=None, batch_size=64, num_steps=64):
    with open('D:\\data\\training\\translation2019zh\\translation2019zh_train.json', 'r', encoding='utf-8') as f:
        if num_lines is None:
            lines = f.readlines()
        else:
            lines = []
            for i in range(num_lines):
                line = f.readline()
                if line == '':
                    break
                lines.append(line)
    # 每行是一个json，转成dict
    lines = [eval(line) for line in lines]
    lines = [(line['english'], line['chinese']) for line in lines]
    # 每组数据是英文tokens和中文tokens
    line_tokens = [(en.lower().split(), jieba.lcut(zh))for en,zh in lines]
    en_lines = [line[0] for line in line_tokens]
    zh_lines = [line[1] for line in line_tokens]
    # 创建语料库
    en_vocab = Vocab(en_lines)
    zh_vocab = Vocab(zh_lines)
    for line in line_tokens:
        if len(line[0]) >= num_steps:
            line[0][num_steps-1] = '<eos>'
            continue
        if len(line[1]) >= num_steps:
            line[1][num_steps-1] = '<eos>'
            continue
        line[0].append('<eos>')
        line[1].append('<eos>')
        # 填充字符，补齐序列长度
        if len(line[0]) < num_steps:
            line[0].extend(['<pad>'] * (num_steps - len(line[0])))
        if len(line[1]) < num_steps:
            line[1].extend(['<pad>'] * (num_steps - len(line[1])))

    train_data = []
    for i in range(0, len(line_tokens), batch_size):
        batch_en = [(en_vocab.to_array(items[0][:num_steps])) for items in line_tokens[i:min(len(line_tokens), i + batch_size)]]
        batch_zh = [(zh_vocab.to_array(items[1][:num_steps])) for items in line_tokens[i:min(len(line_tokens), i + batch_size)]]
        batch_en_len = [len(en) for en in batch_en]
        batch_zh_len = [len(zh) for zh in batch_zh]
        train_data.append([torch.tensor(batch_en), torch.tensor(batch_en_len), torch.tensor(batch_zh), torch.tensor(batch_zh_len)])
    return train_data, en_vocab, zh_vocab