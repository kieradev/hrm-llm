import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

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
    dim: int = 3072
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50257
    multiple_of: int = 256
    norm_eps: float = 1e-6
    max_seq_len: int = 2048
    # HRM specific
    n_h_layers: int = 6
    n_l_layers: int = 6
    N: int = 2 # num high-level cycles
    T: int = 2 # num low-level timesteps per cycle
    M_max: int = 8 # max segments for ACT
    

class HRM(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        
        self.h_layers = nn.ModuleList()
        for _ in range(params.n_h_layers):
            self.h_layers.append(TransformerBlock(params.n_heads, params.dim, params.dim // params.n_heads, params.dim * 4, params.multiple_of))
            
        self.l_layers = nn.ModuleList()
        for _ in range(params.n_l_layers):
            self.l_layers.append(TransformerBlock(params.n_heads, params.dim, params.dim // params.n_heads, params.dim * 4, params.multiple_of))
            
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.q_head = nn.Linear(params.dim, 2, bias=False) # halt, continue
        
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward_recurrent(self, z_h, z_l, x, freqs_cis):
        """
        Implements the recurrent part of the HRM forward pass.
        
        This method is designed to be called within the deep supervision loop.
        It performs N high-level cycles, each with T low-level timesteps.
        """
        # --- Hierarchical Convergence Loop ---
        # The 'no_grad' context implements the 1-step gradient approximation
        with torch.no_grad():
            for _i in range(self.params.N * self.params.T - 1):
                # Low-level module update
                for layer in self.l_layers:
                    z_l = layer(z_l, freqs_cis)
                z_l = self.norm(z_l)
                # High-level module update (every T steps)
                if (_i + 1) % self.params.T == 0:
                    for layer in self.h_layers:
                        z_h = layer(z_h + z_l, freqs_cis) # Fuse z_l into z_h
                    z_h = self.norm(z_h)
        # --- Final Update with Gradients ---
        for layer in self.l_layers:
            z_l = layer(z_l, freqs_cis)
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
            z_h = torch.zeros_like(x)
            z_l = torch.zeros_like(x)
        else:
            z_h, z_l = z_init

        z_h, z_l = self.forward_recurrent(z_h, z_l, x, freqs_cis)
            
        output = self.output(z_h)
        q_values = self.q_head(z_h.mean(dim=1)) # Pool over sequence for Q-values
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1), ignore_index=-1)
            
        return output, (z_h, z_l), q_values, loss

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