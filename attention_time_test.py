import torch
import einops
import time
from longnet.model_old import DilatedAttention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        
        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)

        self.out = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B, T, d = x.shape
        Q, K, V = self.query(x), self.key(x), self.value(x)

        Q = einops.rearrange(Q, 'B T (n d) -> B n T d', n=self.n_heads)
        K = einops.rearrange(K, 'B T (n d) -> B n T d', n=self.n_heads)
        V = einops.rearrange(V, 'B T (n d) -> B n T d', n=self.n_heads)

        attention = (Q @ K.transpose(-2, -1)) / (self.dim ** 0.5)

        mask = torch.tril(torch.ones(T, T)).to(x)
        attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = torch.nn.functional.softmax(attention, dim=-1)

        out = attention @ V
        out = einops.rearrange(out, 'B n T d -> B T (n d)')
        out = self.out(out)

        return out

B, T, n_heads, d = 32, 1024, 16, 1024
x = torch.randn(B, T, d).cuda()
y = torch.randn(B, T, d).cuda()

attention = MultiHeadAttention(d, n_heads).cuda()
dilated_attention = DilatedAttention(d, n_heads, max_len=1024).cuda()

start = time.time()

print(attention(x).shape)

end1 = time.time()

print(dilated_attention(y).shape)

end2 = time.time()

print("einops attention: ", end1 - start)
print("dilated attention: ", end2 - end1)