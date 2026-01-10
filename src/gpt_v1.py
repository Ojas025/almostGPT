import torch
import torch.nn as nn
from time import time

# hyperparameters
batch_size = 64
context_length = 256
learning_rate = 3e-4
eval_iterations = 200
max_iterations = 5000
eval_interval = 500
n_embed = 384
dropout = 0.2
n_heads = 6
n_layers = 6

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(21)

@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for i in range(eval_iterations):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[i] = loss.item()
            
        output[split] = losses.mean()
    
    model.train()
    return output        

# input
with open("../data/sherlock-holmes.txt", "r", encoding="utf-8") as file:
    text = file.read()
    
def remove_spaces(text: str) -> str:
    """Remove multiple contiguous blank lines"""
    flag_blank_line = False
    processed_text = []

    for line in text.split("\n"):
        if not line.strip():
            if not flag_blank_line:
                processed_text.append('')
                flag_blank_line = True
            else:
                continue
        else:
            flag_blank_line = False
            processed_text.append(line.strip())
            

    processed_text = '\n'.join(processed_text)
    
    return processed_text

text = remove_spaces(text)

# vocabulary
unique_characters = sorted(list(set(text)))
vocab_size = len(unique_characters)

# tokenization
character_integer_mapping = { char:index for index,char in enumerate(unique_characters) }
integer_character_mapping = { index:char for index,char in enumerate(unique_characters) }

def encode(string):
    """
    Encode string
        Input: list of characters
        Output: list of mapped integers
    """
    output = []
    for character in string:
        output.append(character_integer_mapping[character])
    
    return output        

def decode(integers):
    """
    Decode string
        Input: list of integers
        Output: corresponding mapped string
    """
    output = ""
    for integer in integers:
        output += integer_character_mapping[integer]
    
    return output        

# encode dataset
encoded_text = encode(text)
data = torch.tensor(encoded_text, dtype=torch.long, device=device)

# train-test split
n = int(0.9 * len(data))

train_data = data[:n]
validation_data = data[n:]

def get_batch(split: str):
    """Generate a batch of data [batch_size, context_length]"""
    data = train_data if split == "train" else validation_data
    indices = torch.randint(len(data) - context_length, (batch_size,))
    
    x = torch.stack([data[index:index+context_length] for index in indices])
    y = torch.stack([data[index+1:index+context_length+1] for index in indices])
    
    return x.to(device),y.to(device)

class Head(nn.Module):
    """Self-Attention Head"""
    
    def __init__(self, d_head):
        super().__init__()
        
        # reduce the dimensionality of the input embedding
        self.query = nn.Linear(n_embed, d_head, bias=False)
        self.key = nn.Linear(n_embed, d_head, bias=False)
        self.value = nn.Linear(n_embed, d_head, bias=False)
        
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        
        K = self.key(x) # [B,T,d_head]
        Q = self.query(x) # [B,T,d_head]
        V = self.value(x) # [B,T,d_head]
        
        W = K @ Q.transpose(-2,-1) * (C**-0.5) # [B,T,T]
        W = W.masked_fill(self.tril[:T,:T] == 0, -1e9)
        W = torch.softmax(W, dim=1) # [B,T,T]
        W = self.dropout(W)
        
        output = W @ V # [B,T,d_head]
        
        return output

class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads in parallel"""

    def __init__(self, num_heads, d_head):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(d_head) for _ in range(num_heads)])
        
        self.W0 = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        Z = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.W0(self.dropout(Z))
        return output
        
class FeedForward(nn.Module):
    """Simple linear neural network"""

    def __init__(self, n_embed):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)

class Block(nn.Module):
    """Transformer Block (Communication + Computation)"""

    def __init__(self, n_embed, num_heads):
        super().__init__()
        
        head_size = n_embed // num_heads

        self.layer_norm1 = LayerNorm(n_embed)
        self.layer_norm2 = LayerNorm(n_embed)

        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forwad = FeedForward(n_embed)

    def forward(self, x):
        x = self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forwad(self.layer_norm2(x))

        return x

class LayerNorm(nn.Module):
    """Layer Normalization"""
    
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        
        self.eps = eps
        
        # parameters
        self.gamma = torch.ones(dim).to(device)
        self.beta = torch.zeros(dim).to(device)
        
    def __call__(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        
        x = (x - x_mean) / torch.sqrt(x_var + self.eps)
        
        output = self.gamma * x + self.beta
        
        return output
    
    def parameters(self):
        return [self.gamma, self.beta]

class BigramLanguageModel(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        
        # Lookup table for logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # each time step (t), has its own positional embedding
        self.positional_embedding_table = nn.Embedding(context_length, n_embed)

        # Single-Head self attention
        # self.self_attention_head = Head(n_embed)

        # Multi-Head self attention
        # self.self_attention_heads = MultiHeadAttention(4, n_embed // 4)
        
        # self.feed_forward = FeedForward(n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        
        self.layer_norm = LayerNorm(n_embed)

        self.linear = nn.Linear(n_embed, vocab_size)
        
    def forward(self, index, targets=None):
        B,T = index.shape
        # index, targets are of the size (B,T)
        # logits are of the size (B,T,C)
        token_embeddings = self.token_embedding_table(index) # (B,T,n_embed)
        
        positional_embeddings = self.positional_embedding_table(torch.arange(T, device=device)) # (T,n_embed)
        
        positional_embeddings = positional_embeddings.unsqueeze(0) # (B,T,n_embed)
        
        x = token_embeddings + positional_embeddings
        
        # x = self.self_attention_heads(x)
        
        # x = self.feed_forward(x)
        
        x = self.blocks(x)
        
        x = self.layer_norm(x)
        
        logits = self.linear(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # reshape tensors
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            
            loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)
        
        return logits, loss      
    
    def generate(self, index, max_tokens):
        """
        For each input index:
            - forward pass -> logits, loss
            - focus only on the last time step
            - compute probabilities
            - get next index
            - concat this new index to the original tensor, this becomes the new input
            
        index - [B,T]
        max_tokens - int            
        """
        for _ in range(max_tokens):
            # restrict len(time_steps) to context_length 
            idx = index[:, -context_length:]
            logits, loss = self(idx) # [B,T,C]
            logits = logits[:,-1,:] # [B,C]
            probabilities = nn.Softmax(dim=-1)(logits) # [B,C]
            
            # sample an index from the distribution
            # this avoids picking only the top choice, introduces variability of output
            next_index = torch.multinomial(probabilities, num_samples=1)
            index = torch.cat((index, next_index), dim=1) # [B,T+1]
            
        return index
    
model = BigramLanguageModel(n_layers).to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

start = time()
for iteration in range(max_iterations):
    if iteration % eval_interval == 0:
        losses = estimate_loss()
        print(f"iteration {iteration} - train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
    
    # sample a batch
    xb, yb = get_batch("train")
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(f"Training Duration: {time() - start}\n")    
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
output = model.generate(context, max_tokens=500)[0].tolist()

print(decode(output))