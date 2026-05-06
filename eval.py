import json
import wandb
from sklearn.metrics import classification_report, accuracy_score, f1_score
from utils import DEVICE


def evaluate_model(trainer, test_dataset, id2label):
    """
    Evaluate model on test set
    
    Args:
        trainer: Trained Trainer object
        test_dataset: Test dataset
        id2label: Dictionary mapping label IDs to label names
    
    Returns:
        eval_results: Dictionary with evaluation metrics
    """
    
    eval_results = trainer.evaluate()
    return eval_results


def compute_classification_report(trainer, test_dataset, id2label):
    """Generate detailed classification report"""
    
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = [item['labels'].item() for item in test_dataset]
    
    report = classification_report(
        labels,
        preds,
        target_names=list(id2label.values()),
        output_dict=True,
        zero_division=0
    )
    
    return report, preds, labels


def save_evaluation_results(eval_results, report, output_file='eval_results.json'):
    """Save evaluation results to JSON file"""
    
    results = {
        'metrics': eval_results,
        'classification_report': report
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_file}")
    return output_file


def log_to_wandb(eval_results, report, report_file='eval_results.json'):
    """Log final metrics and classification report to W&B"""
    
    wandb.log({
        'final/eval_loss': eval_results.get('eval_loss', 0),
        'final/accuracy': eval_results.get('eval_accuracy', 0),
        'final/f1': eval_results.get('eval_f1', 0),
    })
    
    artifact = wandb.Artifact('eval-report', type='evaluation')
    artifact.add_file(report_file)
    wandb.log_artifact(artifact)
    
    print("Results logged to W&B")


def print_evaluation_summary(eval_results, report):
    """Print evaluation summary to console"""
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"F1 Score: {eval_results.get('eval_f1', 'N/A'):.4f}")
    print("="*50 + "\n")
    
    print("Classification Report:")
    print(json.dumps(report, indent=2))
