import torch
from config import *
from data import get_data
from modules import GPT

train_data, val_data, chars, vocab_size = get_data()

model = GPT()
model.load_state_dict(torch.load('./model/best_GPT.pt', map_location=torch.device('cpu')))

str2int = {char:i for i, char in enumerate(chars)}
int2str = {i:char for i, char in enumerate(chars)}
encode = lambda s: [str2int[c] for c in s]
decode = lambda lst: ''.join([int2str[i] for i in lst])

text = 'đồng ruộng bát ngát'
context = torch.asarray([encode(text)], dtype=torch.long, device=device)
print('text:', text)
print('context:',context)
print('output:', decode(model.generate(context, max_new_tokens=200)[0].tolist()))