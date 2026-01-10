"""
Changes in GPT-2 from the original transformer paper:
    - Layer Normalization moved to the input of each sub-block
    - Additional LayerNorm after final self-attention block
"""

"""
Notes:
    - Use of GeLU instead of ReLU to deal with dead activations, as GeLU always contributes a local gradient
    - Includes Flash Attention for optimization
    - Includes ugly number optimization
    - Gradient clipping to a norm -> 1.0
    - Learning rate schedule using cosine decay
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
# from torch.utils.cpp_extension import load
from rich.console import Console
from rich.panel import Panel
import tiktoken
from time import time

console = Console()
device = "cuda" if torch.cuda.is_available() else "cpu"

panel = Panel("ProbablyGPT", title="ProbablyGPT", title_align="center", style="bold white on red")
console.print(panel)
console.print(f"Using device: [yellow]{device}[/yellow]")

def load_data():
    with open("../../data/shakespeare.txt", "r") as file:
        text = file.read()
    
    return text        

@dataclass
class GPTConfig:
    """GPT architecture configurations"""
    context_length: int = 1024 # max context length
    vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 byte tokens + 1 <|endoftext|> token 
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embed: int = 768 # embedding dimension

# flash_attention_kernel = load(name="flash_attention", sources=["flash_attention_kernel.cu"])

# def flash_attention(q,k,v,scale):
#     return flash_attention_kernel(q,k,v,scale)

# class FlashAttention(torch.autograd.Function):
#     """Flash attention implementation"""
#     # TODO: implement CUDA kernel

#     @staticmethod
#     def forward(ctx, q, k, v, scaling):
#         ctx.save_for_backward(q,k,v,scaling)
#         output = flash_attention(q,k,v,scaling)
#         return output
    
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         q, k, v, scale = ctx.saved_tensors
        
#         grad_q, grad_k, grad_v = flash_attention_kernel(grad_outputs, q, k, v, scale)
        
#         return grad_q,grad_k,grad_v,None
        
class MaskedSelfAttention(nn.Module):
    """Masked/Causal Self Attention"""

    def __init__(self, config: GPTConfig, dropout=0.2, is_flash_attention=True):
        super().__init__()
        
        self.is_flash_attention = is_flash_attention
        self.config = config
        self.d_head = config.n_embed // config.n_head
        self.scale = self.d_head ** -0.5
        
        self.qkv_proj = nn.Linear(config.n_embed, 3*config.n_embed)
        
        self.register_buffer("tril", torch.tril(torch.ones(config.context_length, config.context_length)))
        
        self.output_proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Q = self.query(x) # [B,T,d_head]
        # K = self.key(x) # [B,T,d_head]
        # V = self.value(x) # [B,T,d_head]
        
        qkv = self.qkv_proj(x) # [B,T,3*C]
        qkv = qkv.view(B,T,self.config.n_head, 3, self.d_head) # [B,T,n_head,3,d_head]
        Q,K,V = qkv[...,0,:], qkv[...,1,:], qkv[...,2,:] # [B,T,n_head,d_head]
        
        if self.is_flash_attention:
            output = nn.functional.scaled_dot_product_attention(Q,K,V,is_causal=True,scale=self.scale)
        else:
            W = (Q @ K.transpose(-2,-1)) * self.scale # [B,T,n_head,T]
            mask = self.tril[:T, :T].unsqueeze(0).unsqueeze(0)
            W = W.masked_fill(mask == 0, torch.finfo(W.dtype).min)
            W = torch.softmax(W, dim=-1) # apply row-wise
            output = W @ V # [B,T,n_head,d_head]
        
        output = output.reshape(B,T,C)
        return self.output_proj(output)

class LayerNorm(nn.Module):
    """Layern Normalization"""

    def __init__(self, n_embed, eps=1e-5):
        super().__init__()
        
        self.eps = eps
        
        # parameters
        self.gamma = nn.Parameter(torch.ones(n_embed))
        self.beta = nn.Parameter(torch.zeros(n_embed))
        
    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        
        x = (x - x_mean) / torch.sqrt(x_var + self.eps)
        
        output = self.gamma * x + self.beta
        
        return output
    
    def parameters(self):
        return [self.gamma, self.beta]

class FeedForward(nn.Module):
    """Feed forward neural networm"""

    def __init__(self, config: GPTConfig, approximate=False):
        super().__init__()
        
        self.config = config
        
        self.network = nn.Sequential(
            nn.Linear(config.n_embed, 4*config.n_embed),
            nn.GELU(approximate="tanh") if approximate else nn.GELU(),
            nn.Linear(4*config.n_embed, config.n_embed)
        )
        
    def forward(self, x):
        return self.network(x)

class Block(nn.Module):
    """Transformer decoder block"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        
        self.layer_norm_1 = LayerNorm(config.n_embed)
        self.attention = MaskedSelfAttention(config)
        self.layer_norm_2 = LayerNorm(config.n_embed)
        self.feed_forward = FeedForward(config)
        
    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        
        return x


class GPT(nn.Module):
    """Generative Pretrained Transformer"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(config.vocab_size, config.n_embed), # contextual embeddings
            positional_embedding = nn.Embedding(config.context_length, config.n_embed), # positional embeddings
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # decoder blocks
            layer_norm_f = LayerNorm(config.n_embed) # additional layernorm
        ))
        
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # weight sharing between token_embedding and final linear layer
        self.transformer.token_embedding.weight = self.lm_head.weight
    
        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 * (2 * self.config.n_layer) ** -0.5 # scale residuals
            nn.init.normal_(module.weight, mean=0.0, std=std)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, index, targets=None):
        # index: [B,T]
        B,T = index.size()
        
        if T > self.config.context_length:
            console.print(f"[red]Cannot forward context of length {T}[/red]")
            return
        
        pos = torch.arange(0, T, device=device, dtype=torch.long)
        pos_emb = self.transformer.positional_embedding(pos) # [B,T,n_embed]
        
        tok_emb = self.transformer.token_embedding(index) # [B,T,n_embed]
        
        x = tok_emb + pos_emb
        
        for block in self.transformer.blocks:
            x = block(x)
        
        x = self.transformer.layer_norm_f(x) # [B,T,n_embed]
        
        loss = None
        logits = self.lm_head(x) # [B,T,vocab_size]

        if targets is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

# --------------------------------------------------------------------

class DataLoader:
    """Data loader class implementation"""

    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        data = load_data()
        encoder = tiktoken.get_encoding("gpt2")
        tokens = encoder.encode(data)
        self.tokens = torch.tensor(tokens, device=device)
        
        self.index = 0
        
        console.print(f"Loaded {len(self.tokens)} tokens")

    def next_batch(self):
        buffer = self.tokens[self.index:self.index+self.B*self.T+1]
        
        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)
        
        self.index += self.B*self.T
        
        if self.index + (self.B*self.T+1) > len(self.tokens):
            self.index = 0
            
        return x,y            

num_return_sequences = 5
max_length = 50
     
# torch.set_float32_matmul_precision("high")
     
# init model   
model = GPT(GPTConfig(vocab_size=50304)) # remove ugly number 50257 -> 50304
model.eval()
model.to(device)
model.compile()

console.print(f"Initialized model")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
data_loader = DataLoader(B=4,T=1024)

for i in range(1, 51):
    t0 = time()
    x,y = data_loader.next_batch()
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x,y)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    t1 = time()
    dt = (t1-t0)*1000
    tokens_per_sec = (data_loader.B * data_loader.T) / (t1 - t0)
    
    print(f"Step {i} | Loss: {loss.item():.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")


# console.print(f"[green]Tokenized input text[/green]")

# while tokens.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(tokens)
#         logits = logits[:,-1,:] # logits at the last position

#         temperature = 0.9
#         probabilities = torch.softmax(logits / temperature, dim=-1)
        
#         # topk sampling
#         top_k_probs, top_k_indices = torch.topk(probabilities, 50, dim=-1)
#         top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
#         index = torch.multinomial(top_k_probs, 1)
        
#         col = torch.gather(top_k_indices, -1, index)
        
#         # append new col to the sequence
#         tokens = torch.cat((tokens, col), dim=1)
        
# console.print(f"[green]Model trained.\n[/green]")        
        
# for i in range(num_return_sequences):
#     x = tokens[i, :max_length].tolist()
#     decoded = encoding.decode(x)
    
    # print(decoded)