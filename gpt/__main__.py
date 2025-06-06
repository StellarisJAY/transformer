import re
from corpus.vocab import Vocab
import torch
from .model import GPT, GPTConfig
from .trainer import Trainer, TrainerConfig
from torch.utils.data import DataLoader, TensorDataset

def preprocess(content:str, block_size):
    content = content.lower() # 转小写
    # 拆分出句子
    sentences = re.split(r'[.!?]', content) # 按.!?拆分
    # 去掉空格
    sentences = [sentence.strip() for sentence in sentences] # 去掉空格
    # 去掉空句子
    sentences = [sentence for sentence in sentences if sentence] # 去掉空句子

    # tokenize
    tokens = [re.split(r'[ ,\n\t]+', sentence) for sentence in sentences] # 按空格或逗号拆分
    # 去掉空token
    tokens = [[token for token in sentence if token] for sentence in tokens] # 去掉空token
    # 去掉token数量少于4的句子
    tokens = [sentence for sentence in tokens if len(sentence) > 4] # 去掉token数量少于4的句子

    # 构建vocab
    vocab = Vocab(tokens) # 构建vocab

    # 添加结束符，如果句子长度大于block_size，则截断
    for i in range(len(tokens)): # 遍历每个句子
        if len(tokens[i]) > block_size: # 如果句子长度大于block_size，则截断
            tokens[i] = tokens[i][:block_size] # 截断
        tokens[i].append('<eos>') # 添加结束符
    
    # 填充到block_size
    for i in range(len(tokens)): # 遍历每个句子
        if len(tokens[i]) < block_size + 1: # 如果句子长度小于block_size，则填充
            tokens[i] = tokens[i] + ['<pad>'] * (block_size - len(tokens[i]) + 1) # 填充
    # 把token转换成id
    tokens = [vocab.to_array(sentence) for sentence in tokens] # 把token转换成id
    return tokens, vocab

def prepare_data(file: str, block_size=128, batch_size=256, device=None):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    tokens, vocab = preprocess(content, block_size)
    # 构建训练集
    data = torch.tensor(tokens, dtype=torch.long, device=device) # 转换成tensor (num_sentences, block_size)
    # 预测的是下一个token，所以X是前n-1个token，Y是后n-1个token
    X = data[:, :-1] # (num_sentences, block_size-1)
    Y = data[:, 1:] # (num_sentences, block_size-1)
    # 构建dataset
    dataset = TensorDataset(X, Y)
    # 构建dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, vocab


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    block_size = 20
    train_iter, vocab = prepare_data(file='.\\gpt\\short-text.txt', block_size=block_size, device=device, batch_size=batch_size)
    config = GPTConfig(
        vocab_size=len(vocab),
        block_size=block_size,
        num_layers=6,
        num_heads=8,
        d_embd=512,
        d_ff=2048,
        dropout=0.1,
        device=device,
    )
    model = GPT(config)

    trainer = Trainer(
        model=model,
        model_config=config,
        conf=TrainerConfig(
            vocab=vocab,
            num_epochs=30,
            lr=0.003,
        )
    )
    trainer.optimize(train_iter)

    predict_tokens = '''I made a machine which'''.lower().split() # 预测的token
    predict_tokens = vocab.to_array(predict_tokens) # 转换成id
    predict_tokens = torch.tensor(predict_tokens, dtype=torch.long, device=device) # 转换成tensor (block_size,)
    predict_tokens = predict_tokens.view(1, -1) # 转换成 (1, block_size)
    output = model.generate(predict_tokens, max_tokens=20)
    print(output)
    print(' '.join(vocab.to_tokens(output)))
    

