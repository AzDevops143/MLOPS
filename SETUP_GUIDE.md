# MLOps Assignment 2 - Complete Setup Guide

This guide walks you through completing the assignment step-by-step.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.9+ installed
- [ ] Git installed
- [ ] GitHub account
- [ ] Hugging Face account (https://huggingface.co/join)
- [ ] Weights & Biases account (https://wandb.ai/site)
- [ ] Your personal access tokens saved

## Step 1: Get Your Tokens

### Weights & Biases Token
1. Go to https://wandb.ai/authorize
2. Copy your API key
3. Save it somewhere safe

### Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "write" access
3. Save it somewhere safe
4. Go to https://huggingface.co/settings/profile to get your username

## Step 2: Prepare Your Data

You need a CSV file named `data.csv` with at least these columns:

```csv
review,genre
"This book was amazing! I loved the characters.",Science Fiction
"A thrilling mystery that kept me guessing.",Mystery
...
```

If you don't have data, download from:
- Hugging Face Datasets: https://huggingface.co/datasets
- Kaggle: https://kaggle.com/datasets
- UCI ML Repository: https://archive.ics.uci.edu

## Step 3: Clone the Repository

```bash
git clone https://github.com/AzDevops143/Myapp.git
cd Myapp
```

## Step 4: Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your tokens
nano .env  # or use your editor
```

Update with your values:
```
WANDB_API_KEY=your_actual_key
HF_TOKEN=your_actual_token
HF_USERNAME=your_username
```

## Step 5: Install Dependencies

### Option A: CPU (Slower, works on any machine)
```bash
pip install -r requirements.txt
```

### Option B: GPU (Faster, requires CUDA 11.8)
```bash
# For NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then other dependencies
pip install transformers wandb huggingface_hub scikit-learn pandas
```

### Option C: Docker (Recommended for consistency)
```bash
# Make sure you have space
docker system prune -a -f

# Build image (takes 10-15 minutes)
docker build -t mlops-assignment .

# Run training
docker run --rm \
  -e WANDB_API_KEY="your_key" \
  -e HF_TOKEN="your_token" \
  -e HF_USERNAME="your_username" \
  -v $(pwd)/data.csv:/app/data.csv \
  mlops-assignment
```

## Step 6: Understand the Code

Read through these files in order:

1. **utils.py** - Helper functions and dataset class
2. **data.py** - Data loading and preprocessing
3. **train.py** - Training loop with W&B
4. **eval.py** - Evaluation and metrics
5. **main.py** - Orchestrator that ties everything together

## Step 7: Run the Pipeline

```bash
python main.py
```

Expected output:
```
Starting MLOps Assignment 2 Pipeline...
Using device: cuda (or cpu)

1. Loading data...
Loaded 4000 samples

2. Preparing data...
Train samples: 3200, Test samples: 800
Labels: {0: 'Fiction', 1: 'Science Fiction', ...}

...

Pipeline completed successfully!
```

### Debugging Common Issues

If you get errors:

1. **"data.csv not found"** - Place your CSV file in the project root
2. **CUDA out of memory** - Edit CONFIG in main.py, reduce batch_size to 8
3. **WANDB_API_KEY not set** - Check your .env file exists and is correct
4. **Model download fails** - Check internet connection, try again

## Step 8: Monitor Training

While training runs, monitor on:

1. **Console output** - See progress, loss, metrics
2. **W&B Dashboard** - Real-time charts: https://wandb.ai/your-username/mlops-assignment2
3. **Local directory** - Results saved to `results/` folder

## Step 9: Review Results

After training completes:

1. Check `eval_results.json` for metrics
2. View W&B dashboard for training curves
3. Model saved to `model/` directory

Expected metrics (will vary by dataset):
- Accuracy: 0.75-0.95
- F1 Score: 0.75-0.95
- Loss: 0.1-0.5

## Step 10: Push to Hugging Face Hub

If you set HF_TOKEN and HF_USERNAME, model automatically pushes to:
```
https://huggingface.co/your-username/mlops-assignment2-distilbert
```

You can load it later:
```python
from transformers import pipeline

pipe = pipeline(
    'text-classification',
    model='your-username/mlops-assignment2-distilbert'
)

print(pipe('I loved this book!'))
```

## Step 11: Push to GitHub

```bash
# Add all files
git add .

# Commit with message
git commit -m "MLOps Assignment 2: Fine-tuning DistilBERT with W&B tracking"

# Push to GitHub
git push origin main
```

## Step 12: Prepare Report

Write a 4-5 page PDF report covering:

1. **Model Selection** (100-150 words)
   - Why DistilBERT?
   - Advantages over BERT
   - Trade-offs

2. **Training Process** (250-300 words)
   - Data preparation steps
   - Training configuration
   - W&B charts and insights
   - Include screenshot of W&B dashboard

3. **Evaluation Results** (200-250 words)
   - Final metrics (Accuracy, F1, Loss)
   - What do these numbers mean?
   - Classification report analysis
   - Per-class performance

4. **Challenges & Learnings** (200-250 words)
   - What was difficult?
   - How you solved problems
   - What you'd do differently
   - Key MLOps insights gained

## Step 13: Final Submission

Prepare your submission with:

- [ ] GitHub repository link (public)
- [ ] Hugging Face model link (public)
- [ ] W&B dashboard link (public visibility)
- [ ] PDF report (4-5 pages)
- [ ] All links in the report

GitHub repository structure:
```
Myapp/
├── main.py
├── data.py
├── train.py
├── eval.py
├── utils.py
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── README.md
└── eval_results.json (optional)
```

## Key Files Reference

### File: utils.py
Contains:
- GenreDataset class for encoding
- Label mapping functions
- Metrics computation
- Model/tokenizer loading

### File: data.py
Contains:
- Data loading from CSV
- Train/test split with stratification
- Dataset creation
- Data sampling for quick testing

### File: train.py
Contains:
- W&B initialization
- Training arguments setup
- Trainer creation
- Model saving

### File: eval.py
Contains:
- Model evaluation
- Classification report generation
- Results saving
- W&B logging

### File: main.py
Orchestrates all above in order:
1. Load data
2. Prepare data
3. Load model
4. Create datasets
5. Setup W&B
6. Train
7. Evaluate
8. Save results
9. Push to Hub

## Performance Tips

To speed up training:

1. Reduce max_length to 256
2. Reduce batch_size to 8
3. Set samples_per_label to 100 for quick testing
4. Use GPU instead of CPU

To get better results:

1. Use more training data
2. Increase num_epochs to 5
3. Lower learning_rate to 2e-5
4. Increase warmup_steps to 500

## Testing Your Setup

Before full training, test with:

```python
# In main.py CONFIG:
CONFIG = {
    ...
    'samples_per_label': 50,  # Only 50 per label
    'num_epochs': 1,          # Just 1 epoch
    ...
}
```

This completes in minutes and confirms everything works.

## Troubleshooting Commands

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check WANDB connection
wandb login  # Re-authenticate if needed

# Check HF connection
huggingface-cli login  # Re-authenticate if needed

# View W&B logs
wandb offline  # For offline mode

# Clear Docker cache
docker system prune -a -f
```

## Additional Resources

- Hugging Face Course: https://huggingface.co/course
- W&B Docs: https://docs.wandb.ai
- Transformers Docs: https://huggingface.co/docs/transformers
- DistilBERT Paper: https://arxiv.org/abs/1910.01108

Good luck with your assignment!
