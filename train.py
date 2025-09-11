import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import TransformerModel
from data import get_batch
from config import device, block_size

# training hyperparameters
max_iters = 5000  # total steps
eval_interval = 500  # how often evaluate
learning_rate = 3e-4  # base learning rate
warmup_steps = 200  # warmup steps for scheduler
grad_clip = 1.0  # clip gradient size

# create model and optimizer
model = TransformerModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# learning rate scheduler function
def lr_lambda(step):
    if step < warmup_steps:  # linear warmup
        return step / warmup_steps
    progress = (step - warmup_steps) / (max_iters - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))  # cosine decay

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_losses, val_losses = [], []  # store losses

# function to evaluate train and val loss
def evaluate():
    model.eval()
    losses = {"train": 0, "val": 0}
    with torch.no_grad():
        for split in ["train", "val"]:
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[split] = loss.item()
    model.train()
    return losses

# function to generate text from model
def generate(model, start, length=200, temperature=1.0):
    model.eval()
    idx = torch.tensor([[start]], dtype=torch.long, device=device)  # starting char id
    out = idx
    for _ in range(length):
        logits, _ = model(out[:, -block_size:])  # run through model
        logits = logits[:, -1, :] / temperature  # take last step logits scale by temperature
        probs = torch.softmax(logits, dim=-1)  # convert to probabilities
        next_id = torch.multinomial(probs, num_samples=1)  # sample one
        out = torch.cat([out, next_id], dim=1)  # add to output
    model.train()
    return out[0].tolist()

# main training loop
for step in range(max_iters):
    xb, yb = get_batch('train')  # get batch
    logits, loss = model(xb, yb)  # forward pass
    optimizer.zero_grad()  # reset gradients
    loss.backward()  # backward pass
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # clip grads
    optimizer.step()  # update weights
    scheduler.step()  # update learning rate

    if step % eval_interval == 0:
        losses = evaluate()  # eval losses
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        print(f"Step {step}: Train {losses['train']:.3f}, Val {losses['val']:.3f}")

        # save checkpoint
        torch.save(model.state_dict(), f"checkpoint_{step}.pt")

        # generate sample text
        from data import itos
        sample_ids = generate(model, start=0, length=300, temperature=0.8)
        sample_text = ''.join([itos[i] for i in sample_ids])
        print("---- Sample ----")
        print(sample_text)
        print("----------------")

# plot loss curves
plt.plot(range(0, max_iters, eval_interval), train_losses, label="train")
plt.plot(range(0, max_iters, eval_interval), val_losses, label="val")
plt.legend()
plt.xlabel("steps")
plt.ylabel("loss")
plt.savefig("loss_curve.png")
plt.show()