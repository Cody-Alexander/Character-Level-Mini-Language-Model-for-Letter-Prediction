import torch

vocab_file = "input.txt" #use tiny shakespeare text file
block_size = 128 #how many characters context window
batch_size = 64 #how many sequences in one batch
embed_dim = 128 #size of embedding vectors
n_heads = 4 #number of attention heads
n_layers = 2 #number of transformer blocks
dropout = 0.1 #dropout rate to stop overfitting
device = "cuda" if torch.cuda.is_available() else "cpu" #pick gpu if it exists else cpu

# read the whole text file into memory
with open(vocab_file, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))#get all unique characters sorted
vocab_size = len(chars)#size of vocabulary how many unique chars