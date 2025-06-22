import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys

def evaluate_transformer(trainer, train_dataset, test_dataset, y_train, y_test, model, save_dir=None):
    if save_dir is None:
        notebook_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        model_tag = getattr(model.config, "name_or_path", "model")
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        save_dir = f"../../evaluation/{notebook_name}_{model_tag}_eval_{timestamp}"

    os.makedirs(save_dir, exist_ok=True)

    print("EVALUATING ON TEST SET")
    print("Making predictions on test set...")
    test_predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(test_predictions.predictions, axis=1)

    print("Making predictions on training set for comparison...")
    train_predictions = trainer.predict(train_dataset)
    y_train_pred = np.argmax(train_predictions.predictions, axis=1)

    target_names = ['Neutral (0)', 'Bullish (1)', 'Bearish (2)']

    # --- TEST SET PERFORMANCE ---
    print("\n--- TEST SET PERFORMANCE ---")
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = report_dict['macro avg']['precision']
    test_recall = report_dict['macro avg']['recall']
    test_f1 = report_dict['macro avg']['f1-score']

    summary_df = pd.DataFrame({
        'Precision (Macro Avg)': [test_precision],
        'Recall (Macro Avg)': [test_recall],
        'F1-Score (Macro Avg)': [test_f1],
        'Overall Accuracy': [test_accuracy]
    }, index=['FinBERT + LoRA'])

    print(summary_df)
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # --- TRAINING SET PERFORMANCE ---
    print("\n--- TRAINING SET PERFORMANCE ---")
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Difference (Train - Test): {train_accuracy - test_accuracy:.4f}")

    if train_accuracy - test_accuracy > 0.1:
        print("Warning: Large gap between train and test accuracy may indicate overfitting")

    # --- CONFUSION MATRIX ---
    print("\n--- CONFUSION MATRIX (TEST SET) ---")
    cm = confusion_matrix(y_test, y_pred)
    class_labels = ['Neutral', 'Bullish', 'Bearish']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - FinBERT + LoRA')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # --- TRAINING HISTORY ---
    print("\n--- TRAINING HISTORY ---")
    train_history = trainer.state.log_history
    train_losses = [entry['loss'] for entry in train_history if 'loss' in entry]
    steps = list(range(len(train_losses)))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, 'b-', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    print(f"\nFinal training loss: {train_losses[-1]:.4f}")

    # --- SAVE RESULTS ---
    print("\n--- SAVING RESULTS ---")
    summary_df.to_csv(os.path.join(save_dir, "summary_metrics.csv"))
    np.save(os.path.join(save_dir, "confusion_matrix.npy"), cm)

    results_dict = {
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm.tolist(),
        'training_samples': len(y_train),
        'test_samples': len(y_test)
    }

    with open(os.path.join(save_dir, "detailed_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("Detailed results saved.\nEvaluation complete!")