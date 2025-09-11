# Character-Level-Mini-Language-Model-for-Letter-Prediction
This project is a character-level transformer (mini language model) built entirely from scratch for my AIML339. The model is trained on the Tiny Shakespeare dataset (approximately one million characters) to predict the next character in a sequence and generate Shakespeare-like text. It follows a GPT-style decoder-only architecture with embeddings, masked self-attention, feedforward layers, and residual connections.

---

## Requirements

- Python 3.10
- PyTorch 2.2+
- NumPy
- Matplotlib
- tqdm
- seaborn

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/char-transformer.git
   cd char-transformer
   ````

2. **Install Dependencies:**

   ```bash
   pip install torch numpy matplotlib tqdm seaborn
   ```

3. **Download the Dataset:**

   ```bash
   curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
   ```

   Make sure `input.txt` is in the project root folder.

---

## Usage

### Training

Run the training script to train the model:

```bash
python train.py
```

This will:

* Train the transformer on Tiny Shakespeare
* Save checkpoints 
* Log train/validation loss
* Generate text samples during training
* Save a loss curve plot

### Generate Text

Text samples are printed during training. You can also use the `generate()` function in `train.py` to create custom samples after training.

### Visualize Attention

Run the visualization script to produce attention heatmaps:

```bash
python visualize.py
```

This will save heatmaps.

---

## Troubleshooting

* **Import Errors:** Make sure dependencies are installed. 
* **CUDA Not Available:** The code will fall back to CPU automatically. Training will be slower.
* **High Memory Use:** Reduce `block_size`, `embed_dim`, or `batch_size` in `config.py`.
* **NaN Loss:** Ensure gradient clipping and warmup scheduler are active (already included).

---

## Acknowledgments

* **Dataset:** [Tiny Shakespeare by Karpathy](https://github.com/karpathy/char-rnn)
* **Framework:** PyTorch
