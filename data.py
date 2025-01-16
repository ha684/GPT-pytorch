from datasets import load_dataset
import torch
from config import *

def get_data():
    # load dataset from huggingface
    dataset = load_dataset('truongpdd/vietnamese_poetry')
    
    # join all text in the dataset
    text = '\n'.join(dataset['train'][:]['text'])
    
    # get all unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # encode and decode functions
    str2int = {char:i for i, char in enumerate(chars)}
    int2str = {i:char for i, char in enumerate(chars)}
    encode = lambda s: [str2int[c] for c in s]
    decode = lambda lst: ''.join([int2str[i] for i in lst])
    
    # convert data from string to number
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # split data into train and validation
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    # block_size = block_size
    # x_train = train_data[:block_size]
    # y_train = train_data[1:block_size+1]
    # for i in range(block_size):
    #     print(f'context: {x_train[:i+1]} => target: {y_train[i]}')
    return train_data, val_data, chars, vocab_size


def get_batch(split='train'):
    data =  get_data()[0] if split == 'train' else get_data()[1]
    start_idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in start_idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in start_idx])
    x = x.to(device)
    y = y.to(device)
    return x, y

if __name__=='__main__':
    x, y = get_batch()
    print(x.shape, y.shape)
        