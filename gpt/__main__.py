import re
from corpus.vocab import Vocab
import torch
from .model import GPT, GPTConfig
from .trainer import Trainer, TrainerConfig
from torch.utils.data import DataLoader, TensorDataset
from .plot import plot_attention_weights

def preprocess(content:str, block_size):
    content = content.lower() # 转小写
    # 所有标点符号转换成空格
    content = re.sub(r'[^\w\s]', ' ', content)
    # 按空格 \n \t分词
    tokens = re.split(r'[ \n\t]+', content)

    # 去掉空的token
    tokens = [token for token in tokens if token]

    # 按每组block_size-1个token分组
    tokens = [tokens[i:i+block_size-1] for i in range(0, len(tokens), block_size-1)]
    # 每组末尾增加<eos>token
    tokens = [sentence + ['<eos>'] for sentence in tokens]

    tokens = tokens[:-1]

    # 构建vocab
    vocab = Vocab(tokens) # 构建vocab
    
    # 填充到block_size
    for i in range(len(tokens)): # 遍历每个句子
        if len(tokens[i]) < block_size: # 如果句子长度小于block_size，则填充
            tokens[i] = tokens[i] + ['<pad>'] * (block_size - len(tokens[i])) # 填充
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
    batch_size = 32
    block_size = 128
    train_iter, vocab = prepare_data(file='.\\gpt\\timemachine.txt', block_size=block_size, device=device, batch_size=batch_size)
    config = GPTConfig(
        vocab_size=len(vocab),
        block_size=block_size,
        num_layers=2,
        num_heads=4,
        d_embd=768,
        d_ff=3072,
        dropout=0.0,
        device=device,
    )
    model = GPT(config)

    trainer = Trainer(
        model=model,
        model_config=config,
        conf=TrainerConfig(
            vocab=vocab,
            num_epochs=1,
            lr=1.5e-4,
        )
    )
    trainer.optimize(train_iter)

    predict_tokens = '''I made a machine which'''.lower().split() # 预测的token
    predict_tokens = vocab.to_array(predict_tokens) # 转换成id
    predict_tokens = torch.tensor(predict_tokens, dtype=torch.long, device=device) # 转换成tensor (block_size,)
    predict_tokens = predict_tokens.view(1, -1) # 转换成 (1, block_size)
    output = model.generate(predict_tokens, max_tokens=20, end_token=vocab.to_array(["<eos>"])[0])
    print(output)
    output_tokens = vocab.to_tokens(output)
    print(' '.join(output_tokens))

    plot_attention_weights(model=model, tokens=output_tokens, max_layer_heads=1, max_layers=1)
    

