# Character-Level-Mini-Language-Model-for-Letter-Prediction
This project is a character-level transformer (mini language model) built entirely from scratch for my AIML339. The model is trained on the Tiny Shakespeare dataset (approximately one million characters) to predict the next character in a sequence and generate Shakespeare-like text. It follows a GPT-style decoder-only architecture. It explores attention mechanisms, autoregressive learning, and text generation under resource constraints.
The implementation includes:

-A fully custom transformer decoder (model.py)

-Training pipeline with checkpointing and logging (train.py)

-Evaluation and visualization tools for loss, accuracy, and perplexity (eval.py, visualize.py)

-Sampling utilities for text generation with temperature control (sample.py)

- ablation and fine-tuning tools (ablation_runner.py, compare_runs.py)
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
   git clone https://github.com/Cody-Alexander/Character-Level-Mini-Language-Model-for-Letter-Prediction.git
   cd Character-Level-Mini-Language-Model-for-Letter-Prediction
   ````

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
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

###Continue Training 

You can resume from a checkpoint:

```bash
   python train.py --resume checkpoints/base_run.pt
```

###Evaluate Checkpoints

Compute cross-entropy loss, token accuracy, and perplexity:
```bash
   python eval.py --checkpoint checkpoints/finetune_run.pt
```

###Compare Runs

To visualize base vs fine-tuned performance:
```bash
   python compare_runs.py
```

###Visualize Results
Plot training and validation curves:
```bash
python visualize.py
```
All generated figures are automatically.

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
