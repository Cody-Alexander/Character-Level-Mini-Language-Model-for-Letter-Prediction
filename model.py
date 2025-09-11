import torch
import torch.nn as nn
from config import vocab_size, block_size, embed_dim, n_heads, n_layers, dropout, device

# one attention head
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, return_attention=False):
        super().__init__()
        # linear layers for key query value
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        # tril makes mask to block looking ahead
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.return_attention = return_attention  # if true store weights
        self.last_attention = None  # storage for attention weights

    def forward(self, x):
        B, T, C = x.shape  # batch length channels
        k = self.key(x)  # make keys
        q = self.query(x)  # make queries
        # compute raw attention weights
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        # mask out future positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # softmax normalize
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # make values
        out = wei @ v  # weighted sum
        if self.return_attention:
            self.last_attention = wei.detach().cpu()  # save for visualization
        return out

# many heads together
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size, return_attention=True) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)  # final projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concat outputs
        out = self.dropout(self.proj(out))
        return out

# feedforward network inside transformer block
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # expand size
            nn.ReLU(),  # nonlinearity
            nn.Linear(4 * embed_dim, embed_dim),  # project back down
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# one block attention + feedforward
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = embed_dim // n_heads  # size per head
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(embed_dim)  # norm before attention
        self.ln2 = nn.LayerNorm(embed_dim)  # norm before feedforward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # residual connection
        x = x + self.ffwd(self.ln2(x))  # residual again
        return x

# full transformer model
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)  # token embeddings
        self.position_embedding = nn.Embedding(block_size, embed_dim)  # positional embeddings
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])  # stack blocks
        self.ln_f = nn.LayerNorm(embed_dim)  # final norm
        self.head = nn.Linear(embed_dim, vocab_size)  # projection to vocab size

    def forward(self, idx, targets=None):
        B, T = idx.shape  # batch size seq length
        tok_emb = self.token_embedding(idx)  # lookup embeddings
        pos = torch.arange(T, device=device)  # positions 0 to T-1
        pos_emb = self.position_embedding(pos)[None, :, :]  # add batch dimension
        x = tok_emb + pos_emb  # combine
        x = self.blocks(x)  # pass through transformer blocks
        x = self.ln_f(x)  # norm
        logits = self.head(x)  # final predictions

        if targets is None:  # if no labels given just return predictions
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B*T, C)  # flatten
        targets = targets.view(B*T)
        loss = nn.functional.cross_entropy(logits, targets)  # loss function
        return logits, loss