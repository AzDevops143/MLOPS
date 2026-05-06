#main execution
import os
import wandb
from dotenv import load_dotenv

from data import load_data, prepare_data, create_datasets, sample_data
from utils import load_tokenizer_and_model, DEVICE, get_env_variable
from train import setup_wandb, train_model, save_model_locally
from eval import evaluate_model, compute_classification_report, save_evaluation_results, log_to_wandb, print_evaluation_summary

load_dotenv()


def main():
    """Main pipeline orchestrator"""
    
    print("Starting MLOps Assignment 2 Pipeline...")
    print(f"Using device: {DEVICE}")
    
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
    
    try:
        print("\n1. Loading data...")
        df = load_data(CONFIG['data_file'])
        print(f"Loaded {len(df)} samples")
        
        if CONFIG['samples_per_label']:
            print(f"Sampling {CONFIG['samples_per_label']} samples per label...")
            df = sample_data(df, CONFIG['samples_per_label'])
        
        print("\n2. Preparing data...")
        train_texts, train_labels, test_texts, test_labels, label2id, id2label = prepare_data(
            df,
            CONFIG['text_column'],
            CONFIG['label_column'],
            test_size=CONFIG['test_size']
        )
        print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
        print(f"Labels: {id2label}")
        
        print("\n3. Loading tokenizer and model...")
        tokenizer, model = load_tokenizer_and_model(
            CONFIG['model_name'],
            num_labels=len(label2id),
            device=DEVICE
        )
        print(f"Loaded {CONFIG['model_name']} with {len(label2id)} labels")
        
        print("\n4. Creating datasets...")
        train_dataset, test_dataset = create_datasets(
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            tokenizer,
            max_length=CONFIG['max_length']
        )
        print(f"Datasets created successfully")
        
        print("\n5. Setting up Weights & Biases...")
        setup_wandb(
            project_name='mlops-assignment2',
            run_name='distilbert-fine-tuning',
            model_name=CONFIG['model_name'],
            hyperparameters={
                'epochs': CONFIG['num_epochs'],
                'batch_size': CONFIG['batch_size'],
                'learning_rate': CONFIG['learning_rate'],
                'max_length': CONFIG['max_length'],
                'model': CONFIG['model_name']
            }
        )
        
        print("\n6. Training model...")
        trainer, training_result = train_model(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=CONFIG['output_dir'],
            num_epochs=CONFIG['num_epochs'],
            batch_size=CONFIG['batch_size'],
            learning_rate=CONFIG['learning_rate'],
            warmup_steps=CONFIG['warmup_steps'],
            logging_steps=CONFIG['logging_steps']
        )
        print("Training completed")
        
        print("\n7. Evaluating model...")
        eval_results = evaluate_model(trainer, test_dataset, id2label)
        report, preds, labels = compute_classification_report(trainer, test_dataset, id2label)
        print_evaluation_summary(eval_results, report)
        
        print("\n8. Saving results...")
        results_file = save_evaluation_results(eval_results, report)
        log_to_wandb(eval_results, report, results_file)
        
        print("\n9. Saving model locally...")
        save_model_locally(model, tokenizer, CONFIG['model_output_dir'])
        
        print("\n10. Pushing to Hugging Face Hub...")
        hf_token = get_env_variable('HF_TOKEN', None)
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                model_repo = 'distilbert-goodreads-genres'
                model.push_to_hub(model_repo)
                tokenizer.push_to_hub(model_repo)
                hf_url = f"https://huggingface.co/{os.getenv('HF_USERNAME', 'username')}/{model_repo}"
                wandb.run.summary['huggingface_model'] = hf_url
                print(f"Model pushed to {hf_url}")
            except Exception as e:
                print(f"Note: Could not push to Hub: {e}")
                print("Set HF_TOKEN and HF_USERNAME environment variables to enable this")
        else:
            print("Set HF_TOKEN environment variable to push model to Hub")
        
        wandb.finish()
        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        wandb.finish()
        raise


if __name__ == '__main__':
    main()
