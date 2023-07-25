import torch
from model import DilatedAttention
from transformers import AutoTokenizer

### hyperparameter

max_len = 4096

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", eos_token='</s>', pad_token='<pad>', model_max_length=max_len)
vocab_size = tokenizer.vocab_size + 3
n_layers = 12
n_heads = 16
dim = 1024
f_dim = 4096

num_epochs = 5
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-2

model = DilatedAttention(dim, n_heads, max_len).cuda()

# example_sentence = "</s> The quick brown fox jumps over the lazy dog. </s>"
# input_ids = tokenizer(example_sentence, padding='max_length',
#                       max_length=max_len, truncation=True, return_tensors='pt').input_ids.long().unsqueeze(0).cuda()

x = torch.randn(batch_size, max_len, dim).cuda()
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    y = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total"))
print(y.shape)