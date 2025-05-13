from ..transformer.transformer import create_transformer, train_transformer, transformer_predict, load_params
from .en_zh_mini import load_en_zh_data
from .enzh2019 import load_train_data

def run_enzh2019_translator(lr, num_epochs, train:bool, predict:bool, input:str, params_file:str):
    num_steps = 32
    num_hiddens = 32
    num_heads = 4
    ffn_num_inputs = 32
    ffn_num_hiddens = 64
    norm_shape = [32]
    enc_layers, dec_layers = 2, 2
    q_size, k_size, v_size = 32,32,32
    train_data, src_vocab, tgt_vocab = load_train_data(num_lines=1024, num_steps=num_steps)
    encoder, decoder, model = create_transformer(len(src_vocab), len(tgt_vocab), 
                                                       num_steps, q_size, k_size, v_size, 
                                                       num_hiddens, num_heads, norm_shape, 
                                                       ffn_num_inputs, ffn_num_hiddens, 
                                                       enc_layers, dec_layers, 
                                                       dropout=0.1)
    train_transformer(model, train_data, tgt_vocab, lr, num_epochs) 

    test_en = ['']
    for X in test_en:
        Y = transformer_predict(model, X, src_vocab, tgt_vocab, delim='')
        print(f'{X}->{Y}')

def run_en_zh_mini(lr, num_epochs, train:bool, predict:bool, input:str, params_file:str):
    num_steps = 32
    num_hiddens = 32
    num_heads = 4
    ffn_num_inputs = 32
    ffn_num_hiddens = 64
    norm_shape = [32]
    enc_layers, dec_layers = 2, 2
    q_size, k_size, v_size = 32,32,32

    train_data, src_vocab, tgt_vocab = load_en_zh_data(batch_size=64, num_steps=num_steps, num_examples=5000)
    
    encoder, decoder, model = create_transformer(len(src_vocab), len(tgt_vocab), 
                                                       num_steps, q_size, k_size, v_size, 
                                                       num_hiddens, num_heads, norm_shape, 
                                                       ffn_num_inputs, ffn_num_hiddens, 
                                                       enc_layers, dec_layers, 
                                                       dropout=0.1)
    if params_file is not None and params_file!= '':
        load_params(model, params_file)
    if train:
        train_transformer(model, train_data, tgt_vocab, lr, num_epochs)
    engs = [
        'Hello .',
        'It\'s a dead end .',
        'Please give me something hot to drink .',
        'She didn\'t try to hide the truth .',
        'Industry as we know it today didn\'t exist in those days'
    ]
    for i in range(len(engs)):
        eng = engs[i]
        translation = transformer_predict(model, eng, src_vocab, tgt_vocab, delim='')
        print(f'{eng} -> {translation}')