# scripts/create_balanced_dataset.py
"""
Creates a balanced dataset of financial tweets by holding out a test set
from the real labeled data, and combining the rest with synthetic tweets
(bearish and neutral) to create a balanced train/val dataset.

Requires: fsspec, huggingface_hub
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_balanced_dataset(
    output_trainval="../../dataset/initial_balanced_tweets.csv",
    output_test="../../dataset/test_set.csv",
    test_size=5000,
    random_state=42
):
    try:
        import fsspec
        import huggingface_hub
    except ImportError:
        raise ImportError("Please install fsspec and huggingface_hub to load hf:// Parquet files:\n  pip install fsspec huggingface_hub")

    if os.path.exists(output_trainval) and os.path.exists(output_test):
        print("Both train/val and test datasets already exist. Skipping creation.")
        return

    print("Loading labeled real tweets...")
    df = pd.read_parquet("hf://datasets/TimKoornstra/financial-tweets-sentiment/data/train-00000-of-00001.parquet")
    df.drop(columns=['url'], inplace=True, errors='ignore')

    print("Creating balanced test set...")
    test_per_class = test_size // 3
    test_parts = []
    for label in [0, 1, 2]:
        class_subset = df[df['sentiment'] == label].sample(n=test_per_class, random_state=random_state)
        test_parts.append(class_subset)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=random_state)
    rest_df = df.drop(test_df.index)

    print("Test set class distribution:")
    print(test_df['sentiment'].value_counts())

    print("Loading synthetic tweets...")
    synth_df = pd.read_parquet("hf://datasets/TimKoornstra/synthetic-financial-tweets-sentiment/data/train-00000-of-00001.parquet")

    # Adjust for samples removed into test set
    bearish = synth_df[synth_df['sentiment'] == 2].head(8826)
    neutral = synth_df[synth_df['sentiment'] == 0].head(5187)

    print("Combining training+val set with synthetic samples...")
    trainval_df = pd.concat([rest_df, bearish, neutral], ignore_index=True).sample(frac=1, random_state=random_state)

    print("Train+Val class distribution:")
    print(trainval_df['sentiment'].value_counts())

    os.makedirs(os.path.dirname(output_trainval), exist_ok=True)
    trainval_df.to_csv(output_trainval, index=False)
    test_df.to_csv(output_test, index=False)

    print(f"\nSaved train/val dataset to: {output_trainval}")
    print(f"Saved final gold test set to: {output_test}")