import torch
import einops
import math

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=15000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x):
        # x: [bs, num_attention_heads, seq_len, head_size]
        B, T, d = x.shape

        return (
            self.cos_cached[:, :, :T, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :T, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.squeeze(0).squeeze(0), sin.squeeze(0).squeeze(0)
    print(cos.shape, sin.shape, q.shape, k.shape)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DilatedAttention(torch.nn.Module):
    def __init__(self, dim, n_heads, max_len):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.max_len = max_len

        self.pos_emb = LlamaRotaryEmbedding(dim, max_len)

        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)

        self.out = torch.nn.Linear(dim, dim)
    
    def dilated_attention_once(self, Q, K, dilated_rate):
        B, n_heads, T, d_head = Q.shape
        result = torch.ones(B, n_heads, T, T, device=Q.device) * float('-inf')
        for i in range(n_heads):
            shift = i % dilated_rate
            result[:, i, shift::dilated_rate, shift::dilated_rate] = Q[:, i, shift::dilated_rate, :] @ K[:, i, :, shift::dilated_rate]

        return result

    def dilated_attention(self, Q, K, segment_length, dilated_rate):
        # Q: B, n_heads, T, d_head
        # K: B, n_heads, d_head, T
        B, n_heads, T, d_head = Q.shape
        result = torch.ones(B, n_heads, T, T, device=Q.device) * float('-inf')

        for i in range(T // segment_length + 1):
            result[:, :, i*segment_length:(i+1)*segment_length, i*segment_length:(i+1)*segment_length] = \
                self.dilated_attention_once(Q[:, :, i*segment_length:(i+1)*segment_length, :], K[:, :, :, i*segment_length:(i+1)*segment_length], dilated_rate)

        return result
    
    def dilated_attention_full(self, Q, K):
        B, n_heads, T, d_head = Q.shape
        result = torch.ones(B, n_heads, T, T, device=Q.device) * float('-inf')

        for X in range(int(math.log2(T))):
            segment_length = 2 ** (X+2)
            dilated_rate = 2 ** X
            
            result += self.dilated_attention(Q, K, segment_length=segment_length, dilated_rate=dilated_rate)

        return result

    def forward(self, x):
        B, T, d = x.shape
        print(x.shape)
        Q, K, V = self.query(x), self.key(x), self.value(x)

        cos, sin = self.pos_emb(x)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        Q = einops.rearrange(Q, 'B T (n d) -> B n T d', n=self.n_heads)
        K = einops.rearrange(K, 'B T (n d) -> B n T d', n=self.n_heads).transpose(-1, -2)
        V = einops.rearrange(V, 'B T (n d) -> B n T d', n=self.n_heads)

        attention = self.dilated_attention_full(Q, K)

        mask = torch.tril(torch.ones(T, T)).cuda()
        attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = torch.nn.functional.softmax(attention, dim=-1)

        out = attention @ V
        out = einops.rearrange(out, 'B n T d -> B T (n d)')
        out = self.out(out)

        return out

class FeedForwardLayer(torch.nn.Module):
    def __init__(self, dim, f_dim):
        super().__init__()

        self.dim = dim
        self.f_dim = f_dim

        self.key = torch.nn.Linear(dim, f_dim)
        self.value = torch.nn.Linear(f_dim, dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.key(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.value(x)

        return x

class DecoderBlock(torch.nn.Module):
    def __init__(self, dim, f_dim, max_len, n_heads=12):
        super().__init__()

        self.dim = dim
        self.f_dim = f_dim
        self.max_len = max_len

        self.attention = DilatedAttention(dim, n_heads, max_len)
        self.ffn = FeedForwardLayer(dim, f_dim)

        self.ln1 = torch.nn.LayerNorm(dim)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x):
        '''
        x: (B, T, dim)
        '''
        out1 = self.attention(x)
        out1 = self.ln1(x + self.dropout(out1))

        out2 = self.ffn(out1)
        out2 = self.ln2(out1 + self.dropout(out2))

        return out2

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, dim, f_dim, max_len, n_layers, n_heads):
        super().__init__()

        self.emb = torch.nn.Embedding(vocab_size, dim)

        self.dim = dim
        self.f_dim = f_dim
        self.max_len = max_len

        self.layers = torch.nn.ModuleList([
            DecoderBlock(dim, f_dim, max_len, n_heads) for _ in range(n_layers)
        ])

        self.out = torch.nn.Linear(dim, vocab_size)

    def forward(self, x):
        '''
        x: (B, T)
        '''
        B, T = x.shape
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x)

        x = self.out(x)

        return x

    def generate(self, input_ids, device, max_length=64, top_p=0.9):
        
        for i in range(max_length):
            outputs = self(input_ids, device)
            next_token_logits = outputs[0][-1, :]
            
            # top-p sampling
            # apply a softmax to convert the logits to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # sort the probabilities in descending order and compute their cumulative sum
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # sample from the filtered distribution
            next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        
        return input_ids[0]
