from transformer.transformer import create_transformer, train_transformer, transformer_predict, load_params
import os
from d2l import torch as d2l
import argparse

batch_size = 64
num_steps = 10
num_hiddens = 32

num_heads = 4
ffn_num_inputs = 32
ffn_num_hiddens = 64

q_size, k_size, v_size = 32,32,32
enc_num_layers = 2
dec_num_layers = 2
norm_shape = [32]

lr = 0.005
num_epochs = 100

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
 '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def load_data_nmt(batch_size, num_steps, num_examples=6000):
    text = d2l.preprocess_nmt(read_data_nmt())
    source, target = d2l.tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = d2l.build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = d2l.build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def run_d2l(lr, num_epochs, train:bool, predict:bool, input:str, params_file:str):
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=batch_size, num_steps=num_steps)
    
    encoder, decoder, transformer = create_transformer(len(src_vocab), len(tgt_vocab), 
                                                       num_steps, q_size, k_size, v_size, 
                                                       num_hiddens, num_heads, norm_shape, 
                                                       ffn_num_inputs, ffn_num_hiddens, 
                                                       enc_num_layers, dec_num_layers, 
                                                       dropout=0.1, device=d2l.try_gpu())
    if params_file is not None and params_file != '':
        load_params(transformer, params_file)
    if train:
        train_transformer(transformer, train_iter, tgt_vocab, lr, num_epochs)
    
    engs = ['I know .', 'Goodbye .', 'Be still .', 'I trust you .', 'I am sorry .', 'Stay down .', 'I will go .', 'Forget it .']
    for i in range(len(engs)):
        eng = engs[i]
        translation = transformer_predict(transformer, eng, src_vocab, tgt_vocab, delim=' ')
        print(f'{eng} -> {translation}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--train', type=bool, default=False, help='train or not')
    parser.add_argument('--params', type=str, default='', help='param file path')
    parser.add_argument('--save', type=str, default='', help='save param file path')
    parser.add_argument('--predict', type=bool, default=False, help='predict or not')
    parser.add_argument('--input', type=str, default='', help='input text')
    args = parser.parse_args()
    
    print(args)
    run_d2l(args.lr, args.num_epochs, 
            args.train, args.predict, 
            args.input, args.params)