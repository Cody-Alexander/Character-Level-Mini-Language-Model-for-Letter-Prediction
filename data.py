import torch
from config import vocab_file, block_size, batch_size, device

# open dataset file and read text
with open(vocab_file, 'r', encoding='utf-8') as f:
    text = f.read()

# build vocabulary list
chars = sorted(list(set(text)))  # all unique chars
vocab_size = len(chars)  # number of unique chars
stoi = {ch: i for i, ch in enumerate(chars)}  # map char to integer
itos = {i: ch for ch, i in stoi.items()}  # map integer back to char

# function encode string into integers
def encode(s): return [stoi[c] for c in s]
# function decode list of ints back to string
def decode(l): return ''.join([itos[i] for i in l])

# turn all text into tensor of ints
data = torch.tensor(encode(text), dtype=torch.long)
# split into train and validation sets
split_idx = int(0.9 * len(data))  # 90 percent train
train_data = data[:split_idx]
val_data = data[split_idx:]

# function to make one batch of input and target
def get_batch(split):
    src = train_data if split == 'train' else val_data  # choose data
    ix = torch.randint(len(src) - block_size, (batch_size,))  # random start indexes
    # stack slices of length block_size
    x = torch.stack([src[i:i+block_size] for i in ix])
    # targets are input shifted one position
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# test this file if run directly
if __name__ == "__main__":
    xb, yb = get_batch('train')
    print("Input shape:", xb.shape)  # should be batch size by block size
    print("Target shape:", yb.shape)
