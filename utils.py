import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import os
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenreDataset(Dataset):
    """Custom dataset class for genre classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def get_label_maps(unique_labels):
    """Create label to ID and ID to label mappings"""
    label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def compute_metrics(pred):
    """Compute accuracy and F1 score for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def load_tokenizer_and_model(model_name, num_labels, device=DEVICE):
    """Load tokenizer and model from Hugging Face Hub"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(device)
    
    return tokenizer, model


def get_env_variable(var_name, default=None):
    """Safely get environment variables"""
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} not set")
    return value
