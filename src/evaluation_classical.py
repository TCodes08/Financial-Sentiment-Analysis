import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import validation_curve, learning_curve

def evaluate_classical_model(pipeline, X_train, y_train, X_test, y_test, model_name=None, save_dir=None):
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        save_dir = f"../../evaluation/{model_name}_eval_{timestamp}"

    os.makedirs(save_dir, exist_ok=True)

    print("EVALUATING ON TEST SET")
    y_pred = pipeline.predict(X_test)
    target_names = ['Neutral (0)', 'Bullish (1)', 'Bearish (2)']

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
    }, index=[model_name])

    print(summary_df)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\nCONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred)
    class_labels = ['Neutral', 'Bullish', 'Bearish']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    print("\nVALIDATION CURVE")
    C_values = np.logspace(-3, 2, 10)
    train_scores, val_scores = validation_curve(
        pipeline, X_train, y_train,
        param_name='logistic_model__C',
        param_range=C_values,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(C_values, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.semilogx(C_values, val_mean, 'o-', color='red', label='Validation Accuracy')
    plt.fill_between(C_values, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Accuracy Score')
    plt.title(f'Validation Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "validation_curve.png"))
    plt.close()

    print("\nLEARNING CURVE")
    train_sizes, train_scores_lc, val_scores_lc = learning_curve(
        pipeline, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    train_mean_lc = np.mean(train_scores_lc, axis=1)
    train_std_lc = np.std(train_scores_lc, axis=1)
    val_mean_lc = np.mean(val_scores_lc, axis=1)
    val_std_lc = np.std(val_scores_lc, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean_lc, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean_lc - train_std_lc, train_mean_lc + train_std_lc, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean_lc, 'o-', color='red', label='Validation Accuracy')
    plt.fill_between(train_sizes, val_mean_lc - val_std_lc, val_mean_lc + val_std_lc, alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "learning_curve.png"))
    plt.close()

    summary_df.to_csv(os.path.join(save_dir, "summary_metrics.csv"))
    np.save(os.path.join(save_dir, "confusion_matrix.npy"), cm)

    results_dict = {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm.tolist(),
        'training_samples': len(y_train),
        'test_samples': len(y_test)
    }

    with open(os.path.join(save_dir, "detailed_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("\nEvaluation complete. Results saved.")
