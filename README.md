# MLOps Assignment 2: Hugging Face Fine-Tuning & Model Deployment

A complete MLOps pipeline for text classification using DistilBERT, with experiment tracking via Weights & Biases, Docker containerization, and GitHub Actions automation.

## What this project does

- Loads and preprocesses the dataset from `data.csv`
- Fine-tunes a Hugging Face DistilBERT model for genre classification
- Logs metrics and artifacts to Weights & Biases
- Evaluates the model on a test split and saves results locally
- Supports Docker execution and GitHub Actions CI/CD

## Project structure

```
MLOPS/
├── main.py                  # Pipeline orchestration and execution
├── data.py                  # Data loading, sampling, and splitting
├── train.py                 # Training loop using Hugging Face Trainer
├── eval.py                  # Evaluation, metrics, and result saving
├── utils.py                 # Tokenizer/model loading and helper utilities
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── .env.example             # Environment variable template
├── README.md                # Project documentation
├── data.csv                 # Sample dataset used for training
└── .github/workflows/
    └── pipeline.yml        # GitHub Actions workflow
```

## Prerequisites

- Python 3.9+
- Git
- Hugging Face account: https://huggingface.co
- Weights & Biases account: https://wandb.ai
- Optional: GPU for faster training

## Quick setup

### 1. Clone the repository
```bash
git clone https://github.com/AzDevops143/MLOPS.git
cd MLOPS
```

### 2. Configure environment variables
```bash
cp .env.example .env
```

Update `.env` with your own keys:

```ini
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_huggingface_username
```

### 3. Install dependencies

For CPU:
```bash
pip install -r requirements.txt
```

For GPU with CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Running the pipeline locally

```bash
python main.py
```

This will:
- load `data.csv`
- split it into train/test sets
- fine-tune DistilBERT
- evaluate on the test set
- save the model and evaluation outputs

## Running with Docker

Build the image:
```bash
docker build -t mlops-assignment .
```

Run the container:
```bash
docker run --rm \
  -e WANDB_API_KEY="your_key" \
  -e HF_TOKEN="your_token" \
  -e HF_USERNAME="your_username" \
  mlops-assignment
```

With GPU support:
```bash
docker run --rm --gpus all \
  -e WANDB_API_KEY="your_key" \
  -e HF_TOKEN="your_token" \
  -e HF_USERNAME="your_username" \
  mlops-assignment
```

## GitHub Actions workflow

The repository includes `.github/workflows/pipeline.yml`.

### What runs automatically
- `lint-and-test` on every push and pull request
- `build-and-push-docker` on every push and pull request

### What runs only when requested
- `train-and-evaluate` when triggered manually or when a commit message contains `[train]`

### Manual trigger
1. Open GitHub Actions for the repo
2. Select `MLOps Pipeline`
3. Click `Run workflow`
4. Set `run_training` to `true`

### Commit-triggered training
```bash
git commit -m "Improve model [train]"
git push origin main
```

## Configuring the pipeline

Edit the `CONFIG` dictionary in `main.py` to change behavior:

```python
CONFIG = {
    'data_file': 'data.csv',
    'text_column': 'review',
    'label_column': 'genre',
    'model_name': 'distilbert-base-cased',
    'num_epochs': 3,
    'batch_size': 16,
    'learning_rate': 3e-5,
    'max_length': 512,
    'warmup_steps': 100,
    'logging_steps': 50,
    'samples_per_label': None,
    'test_size': 0.2,
    'output_dir': './results',
    'model_output_dir': './model'
}
```

- `samples_per_label`: set a small number for faster experiments
- `test_size`: controls the train/test split
- `output_dir`: where training checkpoints and logs are stored
- `model_output_dir`: where the final model is saved

## Actual results

| Metric | Score |
|--------|-------|
| Accuracy | 0.4000 |
| F1 Score | 0.2333 |
| Eval Loss | 1.5978 |

> These metrics were produced in a quick local run on the included `data.csv` dataset. Training on a larger dataset or with more epochs will change results.

## Experiment tracking

### Weights & Biases
The project logs:
- training loss
- validation loss
- accuracy and F1 score
- hyperparameters
- model artifacts

If you want to view runs, set your own workspace and use the W&B dashboard.

### Local evaluation output
After evaluation, the project saves `eval_results.json` containing metrics and the classification report.

## Hugging Face model publishing

If `HF_TOKEN` and `HF_USERNAME` are set, the script can push the trained model to the Hugging Face Hub.

Example repository path:
```text
https://huggingface.co/<your-username>/distilbert-goodreads-genres
```

Load a published model:
```python
from transformers import pipeline
pipe = pipeline('text-classification', model='your-username/distilbert-goodreads-genres')
print(pipe('I loved this book!'))
```

## Troubleshooting

### Common issues

- **`data.csv` not found**: make sure it is in the repo root and contains `review` and `genre` columns.
- **W&B not logging**: verify `WANDB_API_KEY` is set in `.env` or in your environment.
- **CUDA memory errors**: lower `batch_size` and `max_length` in `main.py`.

### Helpful commands

```bash
echo $WANDB_API_KEY
echo $HF_TOKEN
echo $HF_USERNAME
```

## Why this model?

DistilBERT is chosen for this assignment because it is:
- smaller and faster than full BERT
- well-suited for fine-tuning on text classification
- easy to deploy and experiment with

## Notes for students

This repo is designed to show how an MLOps pipeline works end to end:
- data preprocessing
- model fine-tuning
- evaluation and reporting
- experiment tracking
- containerization
- automated workflows

To understand the code, start with `main.py`, then read `data.py`, `train.py`, and `eval.py`.

## Links

- GitHub repository: https://github.com/AzDevops143/MLOPS
- Hugging Face model: https://huggingface.co/your-username/distilbert-goodreads-genres
- W&B project: https://wandb.ai/your-username/mlops-assignment2
