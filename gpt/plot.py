from matplotlib import pyplot as plt
from .model import GPT
import torch
from corpus.vocab import Vocab

# 注意力权重热力图
def plot_attention_weights(model: GPT, tokens: list[str], vocab: Vocab):
    weights = model.attention_weights()
    layers, heads = len(weights), len(weights[0])
    plt.figure(figsize=(20,20))
    labels = [tokens[i] for i in range(weights[0].shape[1])]
    ticks = [i for i in range(weights[0].shape[1])]
    for i in range(layers):
        for j in range(heads):
            plt.subplot(layers, heads, i * heads + j + 1)
            plt.imshow(weights[i][j].cpu().detach().numpy(), cmap='hot_r', interpolation='nearest')
            plt.xticks(ticks=ticks, labels=labels, rotation=90, rotation_mode="anchor", ha="right")
            plt.yticks(ticks=ticks, labels=labels)
            plt.title(f'L={i+1}, H={j+1}')
            plt.colorbar()
    plt.show()