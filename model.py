import torch
import einops
import xformers.ops
import torchscale.component.xpos_relative_position

from typing import Tuple, List

class DilatedAttention(torch.nn.Module):
    def __init__(self, dim, n_heads, segment_lengths: List[int], dilated_rates: List[int]) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.segment_lengths = segment_lengths
        self.dilated_rates = dilated_rates

        assert len(segment_lengths) == len(dilated_rates)
    
    def forward(self, Q, K, V):
        # Q, K, V: (B, n, T, d)
        B, n, T, d = Q.shape
        out = torch.zeros_like(Q)

        for segment_length, dilated_rate in zip(self.segment_lengths, self.dilated_rates):
            # B: batch size, n: number of heads, d: head dimension
            # L: number of segments, S: segment size
            Q = einops.rearrange(Q, "B n (L S) d -> B n L S d", S=segment_length)
            K = einops.rearrange(K, "B n (L S) d -> B n L S d", S=segment_length)
            V = einops.rearrange(V, "B n (L S) d -> B n L S d", S=segment_length)

            Q_offset = torch.zeros_like(Q[:, :, ::dilated_rate, :, :])
            K_offset = torch.zeros_like(K[:, :, ::dilated_rate, :, :])
            V_offset = torch.zeros_like(V[:, :, ::dilated_rate, :, :])

            # shift for each head
            for head in range(n):
                offset = head % dilated_rate
                
                Q_offset[:, head, :, :, :] = Q[:, head, offset::dilated_rate, :, :]
                K_offset[:, head, :, :, :] = K[:, head, offset::dilated_rate, :, :]
                V_offset[:, head, :, :, :] = V[:, head, offset::dilated_rate, :, :]
                # len(offset::dilated_rate) == T // dilated_rate
                

            Q_offset = einops.rearrange(Q_offset, "B n L S d -> (B L) S n d")
            K_offset = einops.rearrange(K_offset, "B n L S d -> (B L) S n d")
            V_offset = einops.rearrange(V_offset, "B n L S d -> (B L) S n d")

            attn_bias = xformers.ops.fmha.attn_bias.LowerTriangularMask()
            x = xformers.ops.memory_efficient_attention(query=Q_offset, key=K_offset, value=V_offset,
                                                        attn_bias=attn_bias)
            # x: (B L) S n d
            
            x = einops.rearrange(x, "(B L) S n d -> B n L S d", B=B)

            out = einops.rearrange(out, "B n (L S) d -> B n L S d", S=segment_length)
            # x: (B, L, S, n, d), out: (B L S n d)
            
            # for each head
            for head in range(n):
                offset = head % dilated_rate

                out[:, head, offset::dilated_rate, :, :] += x[:, head, :, :, :]

            out = einops.rearrange(out, "B n L S d -> B n (L S) d")

            Q = einops.rearrange(Q, "B n L S d -> B n (L S) d", S=segment_length)
            K = einops.rearrange(K, "B n L S d -> B n (L S) d", S=segment_length)
            V = einops.rearrange(V, "B n L S d -> B n (L S) d", S=segment_length)
    
        return out / len(self.dilated_rates)


class MultiHeadDilatedAttention(torch.nn.Module):
    def __init__(self, dim, n_heads, segment_lengths: List[int], dilated_rates: List[int]) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.segment_lengths = segment_lengths
        self.dilated_rates = dilated_rates

        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)

        self.attention = DilatedAttention(dim, n_heads, segment_lengths, dilated_rates)

        self.out = torch.nn.Linear(dim, dim)
    
    def forward(self, Q, K, V):
        # Q, K, V: (B, T, d)
        Q, K, V = self.query(Q), self.key(K), self.value(V)

        Q = einops.rearrange(Q, "B T (n d) -> B n T d", n=self.n_heads)
        K = einops.rearrange(K, "B T (n d) -> B n T d", n=self.n_heads)
        V = einops.rearrange(V, "B T (n d) -> B n T d", n=self.n_heads)
        # Q, K, V: (B, T, n, d)

        x = self.attention(Q, K, V)  # (B, n, T, d)
        x = einops.rearrange(x, "B n T d -> B T (n d)")

        x = self.out(x)

        return x

class LongNetLayer(torch.nn.Module):
    def __init__(self, dim, n_heads, segment_lengths: List[int], dilated_rates: List[int]) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.segment_lengths = segment_lengths
        self.dilated_rates = dilated_rates

        self.attn = MultiHeadDilatedAttention(dim, n_heads, segment_lengths, dilated_rates)
        self.lm1 = torch.nn.LayerNorm(dim)
        self.lm2 = torch.nn.LayerNorm(dim)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        x_ = self.attn(x, x, x)
        x = self.lm1(x + x_)

        x_ = self.mlp(x)
        x = self.lm2(x + x_)

        return x

class LongNet(torch.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            n_heads: int,
            segment_lengths: List[int],
            dilated_rates: List[int],
            n_layers: int
        ) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.segment_lengths = segment_lengths
        self.dilated_rates = dilated_rates

        self.emb = torch.nn.Embedding(vocab_size, dim)
        self.pos_emb = torchscale.component.xpos_relative_position(dim)

        self.layers = torch.nn.ModuleList([
            LongNetLayer(dim, n_heads, segment_lengths, dilated_rates)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        self.pos_emb = self.pos_emb.to(x.device)