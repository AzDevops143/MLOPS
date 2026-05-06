# MLOps Assignment 2: Hugging Face Fine-Tuning & Model Deployment

A production-ready MLOps pipeline for fine-tuning DistilBERT on text classification tasks with Weights & Biases experiment tracking and Hugging Face Hub deployment.

## Project Overview

This project implements a complete MLOps workflow:

1. **Data Preparation**: Load, sample, and split data with stratification
2. **Model Training**: Fine-tune DistilBERT using Hugging Face Trainer API
3. **Experiment Tracking**: Log all metrics and artifacts to Weights & Biases
4. **Model Evaluation**: Compute accuracy, F1, and generate classification reports
5. **Model Publishing**: Push trained model to Hugging Face Hub
6. **Version Control**: Push all code to GitHub for reproducibility

## Project Structure

```
project/
├── main.py              # Entry point orchestrating the pipeline
├── data.py              # Data loading and preprocessing
├── train.py             # Model training with W&B integration
├── eval.py              # Evaluation and results logging
├── utils.py             # Shared utilities and helper functions
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker containerization
├── .env.example         # Environment variables template
├── README.md            # This file
└── results/             # Training outputs
    └── checkpoint-*/    # Saved checkpoints
```

## Prerequisites

- Python 3.9+
- Git
- Hugging Face account: https://huggingface.co
- Weights & Biases account: https://wandb.ai
- GPU (recommended) or CPU (slower)

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/AzDevops143/MLOPS.git
cd MLOPS
```

### 2. Create Environment Variables
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_huggingface_username
```

### 3. Install Dependencies (CPU)
```bash
pip install -r requirements.txt
```

### 3. Install Dependencies (GPU - CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.0 wandb==0.15.0 huggingface_hub==0.16.1 scikit-learn pandas
```

## Running the Pipeline

### Local Execution
```bash
python main.py
```

### Docker Execution

**Step 1: Clean Docker cache (if disk space is limited)**
```bash
docker system prune -a -f
docker builder prune -a -f
```

**Step 2: Build Docker image**
```bash
docker build -t mlops-assignment .
```

Note: Building may take 10-15 minutes due to PyTorch download. This is normal.

**Step 3: Run container**
```bash
docker run --rm \
  -e WANDB_API_KEY="your_key" \
  -e HF_TOKEN="your_token" \
  -e HF_USERNAME="your_username" \
  mlops-assignment
```

**Step 4: With GPU support**
```bash
docker run --rm --gpus all \
  -e WANDB_API_KEY="your_key" \
  -e HF_TOKEN="your_token" \
  -e HF_USERNAME="your_username" \
  mlops-assignment
```
## GitHub Actions
This repository includes a GitHub Actions workflow at `.github/workflows/pipeline.yml`.

### When the workflow runs
- `lint-and-test` runs on every push and pull request
- `build-and-push-docker` runs on every push/pull request
- `train-and-evaluate` runs only when triggered manually or when a commit message contains `[train]`

### Manual training trigger
1. Go to GitHub → Actions → `MLOps Pipeline`
2. Click `Run workflow`
3. Set `run_training` to `true`
4. Click `Run workflow`

### Commit-triggered training
Use a commit message such as:
```bash
git commit -m "Add model improvements [train]"
git push origin main
```
## Configuration

Edit the `CONFIG` dictionary in `main.py` to customize:

```python
CONFIG = {
    'data_file': 'data.csv',           # Input data file
    'text_column': 'review',           # Column with text
    'label_column': 'genre',           # Column with labels
    'model_name': 'distilbert-base-cased',  # HF model
    'num_epochs': 3,
    'batch_size': 16,
    'learning_rate': 3e-5,
    'max_length': 512,
    'samples_per_label': None,         # Set to N for quick testing
}
```

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.4000 |
| F1 Score | 0.2333 |
| Eval Loss | 1.5978 |

These results were obtained from a short local training run on the supplied `data.csv` dataset.

## Tracking Experiments

### Weights & Biases Dashboard
All metrics are automatically logged:
- Training loss per batch
- Validation loss per epoch
- Accuracy and F1 score
- Learning rate schedule
- GPU/CPU utilization
- All hyperparameters

View your runs: https://wandb.ai/your-username/mlops-assignment2

### Local Results
Evaluation results saved to `eval_results.json`:
```json
{
  "metrics": {
    "eval_loss": 0.1234,
    "eval_accuracy": 0.8765,
    "eval_f1": 0.8750
  },
  "classification_report": {
    "precision": {...},
    "recall": {...},
    "f1-score": {...}
  }
}
```

## Publishing Model

### Hugging Face Hub
Models are automatically pushed if HF_TOKEN is set:
- Repository: `https://huggingface.co/your-username/distilbert-goodreads-genres`
- Includes: Model weights, tokenizer, config

Load published model:
```python
from transformers import pipeline

pipe = pipeline(
    'text-classification',
    model='your-username/distilbert-goodreads-genres'
)

result = pipe('I loved this book!')
print(result)
```

### GitHub
All code pushed to GitHub for reproducibility:
- Scripts: data.py, train.py, eval.py, utils.py, main.py
- Config: requirements.txt, Dockerfile, .env.example
- Documentation: README.md

## Troubleshooting

### Disk Space Error
```bash
docker system prune -a -f
docker builder prune -a -f
```

### CUDA Out of Memory
Reduce batch size or max_length in CONFIG:
```python
CONFIG = {
    'batch_size': 8,      # Reduce from 16
    'max_length': 256,    # Reduce from 512
}
```

### Model Not Found
Ensure `data.csv` is in the project root with correct columns.

### W&B Not Logging
Check WANDB_API_KEY is set:
```bash
echo $WANDB_API_KEY
```

## Model Selection Rationale

DistilBERT was chosen because:
- 40% smaller than BERT with minimal accuracy loss
- 60% faster inference
- Suitable for resource-constrained environments
- Pre-trained on large corpus (Wikipedia + BookCorpus)
- Excellent transfer learning baseline

## Performance Notes

- Training time: 5-15 minutes (GPU), 1-2 hours (CPU)
- Inference speed: ~100 samples/second (GPU), ~5 samples/second (CPU)
- Model size: ~268MB after quantization

## References

- Hugging Face Documentation: https://huggingface.co/docs
- Weights & Biases: https://docs.wandb.ai
- DistilBERT Paper: https://arxiv.org/abs/1910.01108
- Transformers Library: https://github.com/huggingface/transformers

## Links

- Hugging Face Model: https://huggingface.co/your-username/distilbert-goodreads-genres
- W&B Project: https://wandb.ai/your-username/mlops-assignment2
- GitHub Repository: https://github.com/AzDevops143/MLOPS
