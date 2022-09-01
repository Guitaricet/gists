import torch
from einops import rearrange

def multihead_attention(x, W_K, W_Q, W_V, U, n_heads):
    bsz, seq, dim = x.shape
    head_dim = dim // n_heads
    scale = sqrt(W_K.shape[-1])

    k = x @ W_K
    q = x @ W_Q
    v = x @ W_V  # (bs, seq, hid)

    # split heads - process them independently, just like different elements in the batch
    k = rearrange(k, 'bs seq (head k) -> (bs head) seq k', head=n_heads)
    q = rearrange(q, 'bs seq (head k) -> (bs head) seq k', head=n_heads)
    v = rearrange(v, 'bs seq (head k) -> (bs head) seq k', head=n_heads)

    alpha = torch.softmax(k @ q.transpose(1, 2) / scale, dim=-1)  # (bs * head, seq, hid / head) @ (bs / head, hid / head, seq)

    attn = alpha @ v  # (bs * head, seq, seq) @ (bs * head, seq, hid / head)

    attn = rearrange(attn, '(bs head) seq k -> bs seq (head k)', head=self.n_heads)
    attn = attn @ U

    return attn
