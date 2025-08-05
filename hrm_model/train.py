import os
import time
import math
import torch
from model import ModelArgs, HRM

# --- Training Configurations ---
# I/O
out_dir = 'out'
# Data
dataset = 'data'
# Model
init_from = 'scratch' # 'scratch' or 'resume'
# AdamW optimizer
learning_rate = 6e-4
max_iters = 100 # Change for full training
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# --------------------------------

import numpy as np
def get_batch(split, data_dir, block_size, batch_size):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def main():
    
    # --- Model Initialization ---
    model_args = dict(n_layers=12, n_heads=12, dim=3072, vocab_size=50257) # simplified for now
    
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = HRM(gptconf)
    
    model.to(device)
    
    # --- Optimizer ---
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    
    # --- Training Loop ---
    
    # Get the initial batch
    xb, yb = get_batch('train', os.path.join(os.path.dirname(__file__), 'data'), model.params.max_seq_len, 16)
    z = None # Initial hidden state
    
    for iter_num in range(max_iters):
        # timing and logging
        t0 = time.time()
        
        # Deep Supervision Loop
        for step in range(model.params.M_max):
            # forward pass
            logits, z_next, q_values, loss = model(xb, z, yb)
            
            # --- ACT Logic ---
            # Q-learning target
            with torch.no_grad():
                # This is a simplified target for demonstration. A full implementation
                # would use the actual reward (prediction correctness).
                q_target = q_values.max()
            q_loss = F.binary_cross_entropy_with_logits(q_values, torch.ones_like(q_values) * q_target)
            
            # combine losses
            total_loss = loss + q_loss
            
            # backward pass
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            
            # detach hidden state for next segment
            z = (z_next[0].detach(), z_next[1].detach())
            
            # --- Halting Condition (Simplified) ---
            # This is a simplified halting for demonstration. A full implementation
            # would use the stochastic policy from the paper.
            if q_values.argmax() == 0: # halt action
                # Reset batch and hidden state for the next iteration
                xb, yb = get_batch('train', os.path.join(os.path.dirname(__file__), 'data'), model.params.max_seq_len, 16)
                z = None
                break
        
        # timing and logging
        dt = time.time() - t0
        print(f"iter {iter_num}: loss {loss.item():.4f}, q_loss {q_loss.item():.4f}, time {dt*1000:.2f}ms")

if __name__ == '__main__':
    main()