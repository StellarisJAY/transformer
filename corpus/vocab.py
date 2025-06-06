import collections
import torch

class Vocab:
    def __init__(self, line_tokens:list, min_freq=1):
        # line_tokens是二维列表，把他展开成一维
        line_tokens = [token for line in line_tokens for token in line]
        # 统计每个token出现的次数
        token_freqs = collections.Counter(line_tokens)
        self.token_freqs = [('<unk>', min_freq), ('<pad>', min_freq), ('<bos>', min_freq), ('<eos>', min_freq)]
        # 按照出现次数排序
        self.token_freqs.extend(sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)) 
        # 过滤掉出现次数小于min_freq的token
        self.token_freqs = [token_freq for token_freq in self.token_freqs if token_freq[1] >= min_freq]
        self.token_dict = {}
        for i, token_freq in enumerate(self.token_freqs):
            self.token_dict[token_freq[0]] = i
    
    def to_tokens(self, arr: list):
        if isinstance(arr, torch.Tensor):
            arr = arr[0]
        seq =  [self.token_freqs[item][0] if item < self.__len__() else '<unk>' for item in arr]
        return seq

    def to_array(self, tokens: list):
        return [self.token_dict[token] if self.token_dict.__contains__(token) else self.token_dict['<unk>'] for token in tokens]
    
    def __len__(self):
        return len(self.token_freqs)
    
    def __getitem__(self, tokens:list|str):
        if isinstance(tokens, str):
            return self.token_dict[tokens] if self.token_dict.__contains__(tokens) else self.token_dict['<unk>']
        return [self.token_dict[token] if self.token_dict.__contains__(token) else self.token_dict['<unk>'] for token in tokens]