import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math 

import tiktoken

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
# batch_size = 12 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
# block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000 # number of iterations to follow
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

with open('ibong_adarna.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# print('Vocabulary size:', vocab_size)

# # create a mapping from characters to integers
# # text to integer
# stoi = { ch:i for i,ch in enumerate(chars) }
# #integer to text
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

encode = lambda s: enc.encode_ordinary(s)
decode = lambda l: enc.decode(l)


vocab_size = 50304

print('Vocabulary size:', vocab_size)

# Train and test splits
# data = torch.tensor(encode(text), dtype=torch.long)
# vocab_size=len(data.unique())
# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]



n=len(text)
train_data=text[:int(0.9*n)]
val_data=text[int(0.9*n):]

train_data=np.array(encode(train_data), dtype=np.uint64)
val_data=np.array(encode(val_data), dtype=np.uint64)


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # x = torch.stack([data[i:i+block_size] for i in ix])
    # y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # x, y = x.to(device), y.to(device)

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


    return x, y



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    """ combined head and multihead attention classes as per EX1"""
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        #regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout= dropout

        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C= x.size()

        q, k ,v = self.c_attn(x).split(self.n_embd, dim=2)

        k=k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        q=q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v=v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        att= (q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att= att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att= F.softmax(att, dim=-1)
        att= self.attn_dropout(att)
        y= att@v

        y= y.transpose(1,2).contiguous().view(B, T, C)

        y=self.resid_dropout(self.c_proj(y))

        return y 



class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size = n_embd // n_head
        # self.sa = MultiHeadAttention(n_head, head_size)
        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        device=idx.device
        B, T = idx.size()


        # print(f"Max idx: {torch.max(idx)}, Min idx: {torch.min(idx)}")
        # print(f"T: {T}")

        # assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(0,T,dtype=torch.long, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


#running the model itself
model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#old loss was loss>=1.90

#ibong adarna loss 5k iters  loss>=2.5

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print('Generated Text:')
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))


