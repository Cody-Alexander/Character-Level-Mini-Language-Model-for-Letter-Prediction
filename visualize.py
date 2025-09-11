import torch
import matplotlib.pyplot as plt
import seaborn as sns
from model import TransformerModel
from config import device, block_size
from data import encode, decode

# load trained model
model = TransformerModel().to(device)
model.load_state_dict(torch.load("checkpoint_500.pt"))  # load saved checkpoint

# encode a short prompt text into numbers
prompt = "ROMEO:"
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
logits, _ = model(idx)

# get attention weights from first block first head
head = model.blocks[0].sa.heads[0]
attn = head.last_attention[0].numpy()

# plot attention weights as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(attn, cmap="viridis")
plt.title(f"attention heatmap for prompt: {prompt}")
plt.xlabel("key positions")
plt.ylabel("query positions")
plt.savefig("attention_heatmap.png")
plt.show()
