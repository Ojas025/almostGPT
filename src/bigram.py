import torch
import torch.nn as nn

# constants
batch_size = 32
context_length = 8
learning_rate = 1e-3
eval_iterations = 200
max_iterations = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval = 500

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # Lookup table for logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, index, targets=None):
        # index, targets are of the size (B,T)
        # logits are of the size (B,T,C)
        logits = self.token_embedding_table(index)
        
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
            logits, loss = self(index) # [B,T,C]
            logits = logits[:,-1,:] # [B,C]
            probabilities = nn.Softmax(dim=-1)(logits) # [B,C]
            
            # sample an index from the distribution
            # this avoids picking only the top choice, introduces variability of output
            next_index = torch.multinomial(probabilities, num_samples=1)
            index = torch.cat((index, next_index), dim=1) # [B,T+1]
            
        return index
    
model = BigramLanguageModel(vocab_size).to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
output = model.generate(context, max_tokens=500)[0].tolist()

print(decode(output))