import torch
from model import Decoder
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

### hyperparameter

max_len = 256

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

model = Decoder(vocab_size, dim, f_dim, max_len, n_layers, n_heads).cuda()
print("model size: {:_}".format(sum(p.numel() for p in model.parameters())))

tinystories = load_dataset('skeskinen/TinyStories-GPT4')['train']['story'][:1_000_000]

train_dataset = []
for story in tinystories:
    train_dataset.append("{} {} {}".format(tokenizer.eos_token, story, tokenizer.eos_token))
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    torch.save(model.state_dict(), f'./models/model_{epoch}.pt')
    print('Epoch:', epoch)
    
    pbar = tqdm(train_dataloader)
    pbar.set_description(f"epoch: {epoch}, loss: {0.0}")
    for i, batch in enumerate(pbar):
        model.train()
        optimizer.zero_grad()
        tokens = tokenizer(batch, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt').input_ids.long().cuda()

        input_ids = tokens[:, :-1]
        label = tokens[:, 1:]

        outputs = model(input_ids)

        loss = loss_fn(outputs.view(-1, vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()

        pbar.set_description(f"epoch: {epoch}, loss: {loss.item():.3f}")
        
        if i % 100 == 0:
            model.eval()
            input_ids = torch.tensor([tokenizer.encode('</s>')]).cuda()

            print(model.generate(input_ids, 'cuda', max_length=256, top_p=0.9))
        
        if i % 1000 == 0:
            torch.save(model.state_dict(), f'./models/model_checkpoint.pt')

torch.save(model.state_dict(), f'./models/model_final.pt')