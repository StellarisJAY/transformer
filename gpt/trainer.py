from .model import GPT, GPTConfig
from corpus.vocab import Vocab
import torch
from torch import nn
from datetime import datetime

class TrainerConfig:
    def __init__(self, vocab: Vocab, num_epochs, lr):
        self.vocab = vocab
        self.num_epochs = num_epochs
        self.lr = lr

class Trainer:
    def __init__(self, model: GPT, model_config: GPTConfig, conf: TrainerConfig):
        self.model = model
        self.model_config = model_config
        self.conf = conf

    def optimize(self, dataset):
        self.model.train()
        loss = nn.CrossEntropyLoss(reduction='none')
        optim = torch.optim.Adam(lr=self.conf.lr, params=self.model.parameters())
        for epoch in range(self.conf.num_epochs):
            train_losses = []
            start = datetime.now()
            for X,Y in dataset:
                Y = torch.nn.functional.one_hot(Y, num_classes=len(self.conf.vocab)).float()
                Y_pred = self.model(X)
                l = loss(Y_pred, Y).mean()
                optim.zero_grad()
                l.backward()
                optim.step()
                train_losses.append(l)
            print(f"epoch:{epoch+1},train_loss={torch.tensor(train_losses).mean()},time={(datetime.now()-start).total_seconds()}")