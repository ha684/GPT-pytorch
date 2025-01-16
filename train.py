from config import *
import torch
import os
from torch.optim import lr_scheduler
from data import get_batch
from modules import GPT
from tqdm import tqdm

@torch.no_grad()
def estimate_loss_val(model):
    out = {}
    model.eval()
    for split in ['val']:
        losses = torch.zeros(eval_iters).to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters).to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    best_loss = 100000
    if 'best_GPT.py' in os.listdir('./model/'):
        best_model = GPT()
        best_model.load_state_dict(torch.load('./model/best_GPT.pt', map_location=torch.device('cpu')))
        best_loss = estimate_loss_val(best_model)[0]
    
    model = GPT()
    model = model.to(device)
    if 'last_GPT.pt' in os.listdir('.'):
        model.load_state_dict(torch.load('./model/last_GPT.pt', map_location=torch.device('cpu')))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.3)
    
    for i in tqdm(range(max_iters)):
        if i % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            if losses['val'] < best_loss:
                best_loss = losses['val']
                torch.save(model.state_dict(), 'best_GPT.pt')   
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), 'last_GPT.pt') 
        scheduler.step()