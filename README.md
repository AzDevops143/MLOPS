# MLOps Assignment 2: DistilBERT Fine-Tuning Pipeline

This repository contains an end-to-end MLOps pipeline for text classification using a Hugging Face DistilBERT model. The pipeline includes data preparation, training, evaluation, Weights & Biases tracking, Docker support, and GitHub Actions automation.

---

## 🚀 Project overview

The project is designed to teach the full lifecycle of an MLOps workflow:

- **Load and preprocess** a dataset from `data.csv`
- **Fine-tune** a transformer model (`distilbert-base-cased`) for genre classification
- **Evaluate** the model using accuracy, F1 score, and loss
- **Track experiments** with Weights & Biases
- **Save and publish** the trained model
- **Automate** CI/CD with GitHub Actions

---

## 📁 Repository structure

```text
MLOPS/
├── main.py                  # Pipeline orchestrator
├── data.py                  # Data loading and preprocessing
├── train.py                 # Model training and W&B setup
├── eval.py                   # Evaluation and results logging
├── utils.py                 # Utility functions and helpers
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build configuration
├── .env.example             # Environment variable template
├── README.md                # Project documentation
├── data.csv                 # Sample dataset used for training
└── .github/workflows/
    └── pipeline.yml        # CI/CD workflow
```

---

## ✅ Prerequisites

- Python 3.9 or newer
- Git
- Hugging Face account: https://huggingface.co
- Weights & Biases account: https://wandb.ai
- Optional: GPU for faster training

---

## 🛠️ Setup instructions

### 1. Clone the repo

```bash
git clone https://github.com/AzDevops143/MLOPS.git
cd MLOPS
```

### 2. Create your environment file

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```ini
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_huggingface_username
```

> `HF_USERNAME` is your Hugging Face profile name, e.g. `srajam696`.

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

---

## ▶️ Run the pipeline locally

```bash
python main.py
```

This command will:

- load `data.csv`
- split data into training and testing sets
- fine-tune DistilBERT
- evaluate the model
- save evaluation results and the trained model

---

## 🐳 Run with Docker

Build the Docker image:

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

If you have GPU support:

```bash
docker run --rm --gpus all \
  -e WANDB_API_KEY="your_key" \
  -e HF_TOKEN="your_token" \
  -e HF_USERNAME="your_username" \
  mlops-assignment
```

---

## 🧪 GitHub Actions workflow

This repo includes a GitHub Actions workflow in `.github/workflows/pipeline.yml`.

### Automatic jobs
- `lint-and-test` runs on every push and pull request
- `build-and-push-docker` runs on every push and pull request

### Conditional job
- `train-and-evaluate` runs only when manually triggered or when a commit message contains `[train]`

### Manually trigger training
1. Go to GitHub Actions in the repo
2. Select `MLOps Pipeline`
3. Click `Run workflow`
4. Set `run_training` to `true`

### Commit-triggered training

```bash
git commit -m "Improve training [train]"
git push origin main
```

---

## ⚙️ Configuration

The pipeline is driven by the `CONFIG` dictionary in `main.py`.

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

### Helpful tuning tips

- `samples_per_label`: set to a small number for fast experiments
- `test_size`: controls the size of the evaluation split
- `output_dir`: stores checkpoints and logs
- `model_output_dir`: stores the final saved model

---

## 📊 Results from a sample run

| Metric | Score |
|--------|-------|
| Accuracy | 0.4000 |
| F1 Score | 0.2333 |
| Eval Loss | 1.5978 |

> These values were generated from a quick local execution using the sample `data.csv`. Larger datasets and longer training will produce different results.

---

## 📌 Experiment tracking

### Weights & Biases
The project logs:

- training loss
- validation loss
- accuracy
- F1 score
- hyperparameters
- model artifacts

### Local output
After evaluation, results are saved to `eval_results.json`, including the classification report.

---

## 📦 Hugging Face publishing

If `HF_TOKEN` and `HF_USERNAME` are configured, the pipeline can push the trained model to the Hugging Face Hub under your account.

Example publish URL:

```text
https://huggingface.co/YOUR_USERNAME/distilbert-goodreads-genres
```

Load the published model with:

```python
from transformers import pipeline
pipe = pipeline(
    'text-classification',
    model='YOUR_USERNAME/distilbert-goodreads-genres'
)
print(pipe('I loved this book!'))
```

If you do not yet have a Hugging Face account, create one at https://huggingface.co and then update `.env`:

```ini
HF_USERNAME=YOUR_USERNAME
HF_TOKEN=your_huggingface_token
```

---

## 🛠️ Troubleshooting

### Common issues

- `data.csv` not found: verify the file exists in the repository root and includes `review` and `genre` columns.
- W&B not logging: confirm `WANDB_API_KEY` is set correctly.
- CUDA memory errors: reduce `batch_size` and `max_length` in `main.py`.

### Check environment values

```bash
echo $WANDB_API_KEY
echo $HF_TOKEN
echo $HF_USERNAME
```

---

## 💡 Why DistilBERT?

DistilBERT is a great choice for this assignment because it is:

- smaller and faster than full BERT
- easier to fine-tune on moderate datasets
- efficient for model deployment
- well-supported by Hugging Face

---

## 🎓 For students

This repository is built to help you understand how a real MLOps pipeline works:

1. `main.py` orchestrates the workflow
2. `data.py` prepares the dataset
3. `train.py` executes training
4. `eval.py` validates and saves results
5. `utils.py` contains reusable helpers

Start by reading `main.py` and then follow the data and training flow in the other scripts.

---

## 🔗 Useful links

- GitHub repository: https://github.com/AzDevops143/MLOPS
- Hugging Face model: https://huggingface.co/<your-username>/distilbert-goodreads-genres
- W&B project: https://wandb.ai/<your-username>/mlops-assignment2
