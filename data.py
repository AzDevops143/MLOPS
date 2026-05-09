import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import GenreDataset, get_label_maps


def load_data(filepath):
    """Load data from CSV or JSON file"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return df


def prepare_data(df, text_column, label_column, test_size=0.2, random_state=42):
    """
    Prepare and split data into train and test sets
    
    Args:
        df: DataFrame with text and labels
        text_column: Name of column containing text
        label_column: Name of column containing labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_texts, train_labels, test_texts, test_labels, label2id, id2label
    """
    
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    unique_labels = sorted(set(labels))
    label2id, id2label = get_label_maps(unique_labels)
    
    numeric_labels = [label2id[label] for label in labels]
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        numeric_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=numeric_labels
    )
    
    return train_texts, train_labels, test_texts, test_labels, label2id, id2label


def create_datasets(train_texts, train_labels, test_texts, test_labels, 
                   tokenizer, max_length=512):
    """Create GenreDataset objects for training and testing"""
    
    train_dataset = GenreDataset(
        train_texts,
        train_labels,
        tokenizer,
        max_length=max_length
    )
    
    test_dataset = GenreDataset(
        test_texts,
        test_labels,
        tokenizer,
        max_length=max_length
    )
    
    return train_dataset, test_dataset


def sample_data(df, samples_per_label=None):
    """Sample data for faster iteration during development"""
    if samples_per_label is None:
        return df
    
    sampled_df = pd.concat([
        group.sample(min(len(group), samples_per_label), random_state=42)
        for name, group in df.groupby('genre')
    ]).reset_index(drop=True)
    
    return sampled_df
