import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import os
import time

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        
        self.wq = nn.Linear(d_model, n_head * d_head, bias=False)
        self.wk = nn.Linear(d_model, n_head * d_head, bias=False)
        self.wv = nn.Linear(d_model, n_head * d_head, bias=False)
        self.wo = nn.Linear(n_head * d_head, d_model, bias=False)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_head, self.d_head)
        xk = xk.view(bsz, seqlen, self.n_head, self.d_head)
        xv = xv.view(bsz, seqlen, self.n_head, self.d_head)
        
        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # Attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.d_head)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)

class TransformerBlock(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_ffn, multiple_of):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, d_model, d_head)
        self.feed_forward = FeedForward(d_model, d_ffn, multiple_of)
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

from dataclasses import dataclass

@dataclass
class ModelArgs:
    # Model dims tuned for ~117 M params
    dim: int             = 560       # hidden size
    n_layers: int        = 12        # legacy / total blocks (6h + 6l)
    n_h_layers: int      = 6         # high-level blocks
    n_l_layers: int      = 6         # low-level blocks
    n_heads: int         = 8        # must divide dim (560/16=35)
    vocab_size: int      = 50257     # GPT-2 BPE vocab
    ffn_dim: int         = 2304      # 4×560 rounded to 9×256
    multiple_of: int     = 256       # used for rounding in FFN (optional)
    norm_eps: float      = 1e-6
    max_seq_len: int     = 1024
    weight_init: str     = 'lecun_normal'

    # HRM-specific
    N: int               = 4         # high-level cycles
    T: int               = 4         # low-level steps per cycle
    M_max: int           = 8         # ACT segments
    

class HRM(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        
        self.h_layers = nn.ModuleList()
        for _ in range(params.n_h_layers):
            self.h_layers.append(TransformerBlock(params.n_heads, params.dim, params.dim // params.n_heads, params.ffn_dim, params.multiple_of))
            
        self.l_layers = nn.ModuleList()
        for _ in range(params.n_l_layers):
            self.l_layers.append(TransformerBlock(params.n_heads, params.dim, params.dim // params.n_heads, params.ffn_dim, params.multiple_of))
            
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.q_head = nn.Linear(params.dim, 2, bias=False) # halt, continue
        
        freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.register_buffer("freqs_cis", freqs_cis)
        
        # weight init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.params.weight_init == 'lecun_normal':
                # Classic LeCun Normal initialization
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                std = math.sqrt(1.0 / fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            pass

    def forward_recurrent(self, z_h, z_l, x, freqs_cis):
        """
        Implements the recurrent part of the HRM forward pass.
        
        This method is designed to be called within the deep supervision loop.
        It performs N high-level cycles, each with T low-level timesteps.
        """
        z_l_init = z_l.clone() # Save initial L-state

        # --- Hierarchical Convergence Loop ---
        # The 'no_grad' context implements the 1-step gradient approximation
        with torch.no_grad():
            for _i in range(self.params.N * self.params.T - 1):
                # Low-level module update
                l_input = z_l + z_h + x # Condition L-module on H-state and input
                for layer in self.l_layers:
                    z_l = layer(l_input, freqs_cis)
                z_l = self.norm(z_l)
                # High-level module update (every T steps)
                if (_i + 1) % self.params.T == 0:
                    for layer in self.h_layers:
                        z_h = layer(z_h + z_l, freqs_cis) # Fuse z_l into z_h
                    z_h = self.norm(z_h)
                    z_l = z_l_init # Reset L-state for next cycle
        
        # --- Final Update with Gradients ---
        l_input = z_l + z_h + x
        for layer in self.l_layers:
            z_l = layer(l_input, freqs_cis)
        z_l = self.norm(z_l)
        for layer in self.h_layers:
            z_h = layer(z_h + z_l, freqs_cis)
        z_h = self.norm(z_h)
        return z_h, z_l

    def forward(self, tokens: torch.Tensor, z_init=None, targets: torch.Tensor = None):
        x = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[0:x.shape[1]]

        # Initialize hidden states if not provided
        if z_init is None:
            # Truncated normal initialization
            z_h = torch.fmod(torch.randn_like(x), 2)
            z_l = torch.fmod(torch.randn_like(x), 2)
        else:
            z_h, z_l = z_init

        z_h, z_l = self.forward_recurrent(z_h, z_l, x, freqs_cis)
            
        output = self.output(z_h)
        q_logits = self.q_head(z_h.mean(dim=1)) # Pool over sequence for Q-logits
        q_values = torch.sigmoid(q_logits)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1), ignore_index=-1)
            
        return output, (z_h, z_l), q_logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

# --- Training Configurations ---
# I/O
out_dir = 'out'
# Data
dataset = 'data'
batch_size = 8
gradient_accumulation_steps = 4
# Model
init_from = 'scratch' # 'scratch' or 'resume'
# AdamW optimizer
learning_rate = 6e-4
# max_iters should be set based on the dataset size.
# For DSIR-filtered-pile-50M, a full epoch is roughly 4B tokens.
# Calculation: (4 * 10^9 tokens) / (effective_batch_size * seq_len) = iterations
# e.g., (4 * 10^9) / (8 * 1024) = ~488,000 iterations.
max_iters = 2000 # A safe default for initial testing.
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# ACT
act_eps = 0.1
# System
device = 'cuda' # For 4090 GPU
dtype = 'bfloat16' # For 4090 GPU
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
    model_args = dict(
      dim        = 560,
      n_h_layers = 6,
      n_l_layers = 6,
      n_heads    = 8,
      vocab_size = 50257,
      ffn_dim    = 2304,
      max_seq_len= 1024,
      # HRM bits
      N          = 4,
      T          = 4,
      M_max      = 8,
    )
    os.makedirs(out_dir, exist_ok=True)  # ensure checkpoint directory exists
    
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = HRM(gptconf)
    
    model.to(device)
    
    # --- Optimizer ---
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    
    # --- Training Loop ---
    
    # Get the initial batch
    xb, yb = get_batch('train', os.getcwd(), model.params.max_seq_len, batch_size)
    z = None # Initial hidden state
    
    for iter_num in range(max_iters):
        # timing and logging
        t0 = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        # Gradient Accumulation Loop
        for micro_step in range(gradient_accumulation_steps):
            # Get a fresh micro-batch
            xb, yb = get_batch('train', os.getcwd(), model.params.max_seq_len, batch_size)
            z = None # Reset hidden state for each micro-batch

            # Determine the minimum number of segments M_min
            if torch.rand(1).item() < act_eps:
                M_min = torch.randint(2, model.params.M_max + 1, (1,)).item()
            else:
                M_min = 1
                
            q_loss = torch.tensor(0.0, device=device)
            
            # Deep Supervision Loop for one micro-batch
            for step in range(model.params.M_max):
                # forward pass
                logits, z_next, q_logits, loss = model(xb, z, yb)
                
                # Q-learning target
                with torch.no_grad():
                    _, preds = torch.max(logits, -1)
                    # Binary reward: 1 if all tokens in the sequence are correct, 0 otherwise
                    reward = (preds == yb).all(dim=1).float()
                    
                    # Get q_values for next state by applying sigmoid to the logits
                    _, _, q_logits_next, _ = model(xb, z_next, yb)
                    q_values_next = torch.sigmoid(q_logits_next)

                    if step >= model.params.M_max - 1:
                        G_continue = q_values_next[:, 0]
                    else:
                        G_continue = q_values_next.max(dim=1)[0]
                    
                    # G_halt is now a vector of rewards, one for each sample in the batch
                    G_halt = reward
                    q_target = torch.stack([G_halt, G_continue], dim=-1)

                q_loss = F.binary_cross_entropy_with_logits(q_logits, q_target)
                
                # combine losses and scale for accumulation
                total_loss = (loss + q_loss) / gradient_accumulation_steps
                
                # backward pass
                total_loss.backward()
                
                z = (z_next[0].detach(), z_next[1].detach())
                
                # Halt if the majority of the batch votes to halt
                halt_votes = torch.argmax(q_logits, dim=1) == 0
                if step >= M_min - 1 and halt_votes.float().mean() > 0.5:
                    xb, yb = get_batch('train', os.getcwd(), model.params.max_seq_len, batch_size)
                    z = None
                    break
        
        # Update weights after accumulating gradients
        optimizer.step()
        
        # Timing
        dt = time.time() - t0
        # Save checkpoint every 50 iters
        if iter_num % 50 == 0:
            ckpt_path = os.path.join(out_dir, f'checkpoint_{iter_num}.pt')
            torch.save({
                'iteration': iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint at iteration {iter_num} to {ckpt_path}")
        
        # Logging
        total = loss + q_loss
        print(f"iter {iter_num}: ce_loss {loss.item():.4f}, q_loss {q_loss.item():.4f}, total_loss {total.item():.4f}, time {dt*1000:.2f}ms")

        # every 10 iters, do a quick val CE
        if iter_num % 10 == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch('val', os.getcwd(), model.params.max_seq_len, batch_size)
                _, _, _, val_ce = model(vx, None, vy)
                print(f"  → val_iter {iter_num}: val_ce_loss {val_ce.item():.4f}")
            model.train()

if __name__ == '__main__':
    main()
