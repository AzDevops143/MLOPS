import wandb
from transformers import TrainingArguments, Trainer
from utils import compute_metrics, get_env_variable, DEVICE
import os


def setup_wandb(project_name, run_name, model_name, hyperparameters):
    """Initialize Weights & Biases for experiment tracking"""
    
    wandb_api_key = os.getenv('WANDB_API_KEY')
    
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                'model': model_name,
                **hyperparameters
            }
        )
    else:
        print("Warning: WANDB_API_KEY not set. Skipping W&B initialization.")


def train_model(model, train_dataset, test_dataset, output_dir='./results', 
                num_epochs=3, batch_size=16, learning_rate=3e-5,
                warmup_steps=100, logging_steps=50):
    """
    Train the model using Hugging Face Trainer with W&B logging
    
    Args:
        model: Pre-trained model
        train_dataset: Training dataset
        test_dataset: Evaluation dataset
        output_dir: Directory to save results
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Log metrics every N steps
    
    Returns:
        trainer: Trained Trainer object
        training_result: Training results
    """
    
    # Check if W&B is available
    wandb_available = bool(os.getenv('WANDB_API_KEY'))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=logging_steps,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        report_to='wandb' if wandb_available else [],
        run_name='bert-fine-tuning',
        learning_rate=learning_rate,
        seed=42
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    training_result = trainer.train()
    
    return trainer, training_result


def save_model_locally(model, tokenizer, output_path='./model'):
    """Save model and tokenizer locally"""
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")
