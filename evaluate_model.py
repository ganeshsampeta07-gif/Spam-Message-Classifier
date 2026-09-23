import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_and_save_metrics(y_true, y_pred, output_dir='outputs'):
    """
    Calculates evaluation metrics, saves a classification report,
    and plots a confusion matrix.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate core metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='spam')
    rec = recall_score(y_true, y_pred, pos_label='spam')
    f1 = f1_score(y_true, y_pred, pos_label='spam')

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%\n")

    # Generate classification report text
    report = classification_report(y_true, y_pred)
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n")
        f.write(f"Accuracy:  {acc * 100:.2f}%\n")
        f.write(f"Precision: {prec * 100:.2f}%\n")
        f.write(f"Recall:    {rec * 100:.2f}%\n")
        f.write(f"F1 Score:  {f1 * 100:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
        
    print(f"Saved evaluation report to: {report_path}")

    # Generate confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=['ham', 'spam'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Spam', 'Spam'], 
                yticklabels=['Not Spam', 'Spam'])
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Saved confusion matrix plot to: {cm_path}")
