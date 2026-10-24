import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(6))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    labels = [f"G{i+1}" for i in range(6)]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=False)
    ax.set_title(title); ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
