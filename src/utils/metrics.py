"""
Evaluation & Analysis
Handles statistical testing and visualization for the FHE-PHML framework.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_rel

class Evaluator:
    def __init__(self, dataset_name="uci"):
        self.dataset_name = dataset_name

    def calculate_metrics(self, y_true, y_pred):
        """Returns a dictionary of standard ML metrics."""
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1_Score": f1_score(y_true, y_pred),
        }

    def run_t_test(self, plain_correct, fhe_correct):
        """
        Performs a paired t-test to see if FHE significantly 
        differs from Plaintext performance.
        """
        if len(set(plain_correct)) <= 1 and len(set(fhe_correct)) <= 1:
            return None, None
        t_stat, p_value = ttest_rel(fhe_correct, plain_correct)
        return t_stat, p_value

    def plot_comparison(self, plain_metrics, fl_metrics, fhe_metrics, times):
        """Generates the dual-plot visualization for the thesis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Performance Bar Chart
        metrics_to_plot = ['Accuracy', 'F1_Score']
        x = np.arange(len(metrics_to_plot))
        
        axes[0].bar(x - 0.2, [plain_metrics[m] for m in metrics_to_plot], 0.2, label='Plaintext')
        axes[0].bar(x, [fl_metrics[m] for m in metrics_to_plot], 0.2, label='Federated')
        axes[0].bar(x + 0.2, [fhe_metrics[m] for m in metrics_to_plot], 0.2, label='FHE-PHML')

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_to_plot)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_title(f'Performance Comparison ({self.dataset_name.upper()})')
        axes[0].legend()

        # 2. Latency Line Chart
        axes[1].plot(times, marker='o', color='purple', label='Inference Latency')
        axes[1].axhline(y=np.mean(times), color='r', linestyle='--', label='Mean Time')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].set_title('FHE Execution Time per Sample')
        axes[1].legend()

        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
            
        plt.savefig(f"results/results_{self.dataset_name}.png")
        plt.show()
