import torch
import einops
from model import MultiHeadDilatedAttention
import time

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads

        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)

        self.out = torch.nn.Linear(dim, dim)
    
    def forward(self, Q, K, V):
        B, T, d = Q.shape

        Q, K, V = self.query(Q), self.key(K), self.value(V)

        Q = einops.rearrange(Q, "B T (n d) -> B n T d", n=self.n_heads)
        K = einops.rearrange(K, "B T (n d) -> B n T d", n=self.n_heads)
        V = einops.rearrange(V, "B T (n d) -> B n T d", n=self.n_heads)

        out = torch.einsum("b n t d, b n s d -> b n t s", Q, K) / (d ** 0.5)
        mask = torch.tril(torch.ones(T, T)).cuda()
        out = out.masked_fill(mask == 0, float('-inf'))
        out = torch.nn.functional.softmax(out, dim=-1)

        out = torch.einsum("b n t s, b n s d -> b n t d", out, V)
        out = einops.rearrange(out, "B n T d -> B T (n d)")

        out = self.out(out)

        return out

B, T, d = 16, 17, 1024
n_heads = 16

segment_lengths = [4, 8]
dilated_rates = [1, 2]
attn1 = MultiHeadDilatedAttention(d, n_heads, segment_lengths, dilated_rates).cuda() # 4.5GiB
attn2 = MultiHeadAttention(d, n_heads).cuda() # 27GiB

X = torch.randn(B, T, d).cuda()

start = time.time()

out = attn1(X, X, X)
print(out.shape)

end1 = time.time()
print(f"Time taken: {end1 - start:.3f} seconds")

# out = attn2(X, X, X)
# print(out.shape)

end2 = time.time()
print(f"Time taken: {end2 - end1:.3f} seconds")

time.sleep(10)
