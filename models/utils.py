from matplotlib.pyplot import axes
from typing import List, Literal
from dataclasses import dataclass, field
from numpy.typing import NDArray
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, r2_score, roc_auc_score,
                             root_mean_squared_error, mean_absolute_error)


def plot_train_validation_loss_per_epoch(train_loss: List,
                                         validation_loss: List,
                                         ax: axes) -> axes:
    ax.plot(train_loss, label='Train Loss', marker='o')
    ax.plot(validation_loss, label='Validation Loss', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    return ax


@dataclass
class Metrics:
    metric_type: Literal["classification", "regression"]
    y_true: NDArray
    y_pred: NDArray
    results: dict = field(init=False)

    def __post_init__(self):
        self.results = (self.classification()
                        if self.metric_type == "classification"
                        else self.regression())

    def classification(self):
        is_binary = len(set(self.y_true.flatten())) <= 2
        average = "binary" if is_binary else "weighted"
        return {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred,
                                   average=average,
                                   zero_division=0),
            'precision': precision_score(self.y_true, self.y_pred,
                                         average=average, zero_division=0)
        }

    def regression(self):
        return {
            'rmse': root_mean_squared_error(self.y_true, self.y_pred),
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'r2': r2_score(self.y_true, self.y_pred)
        }
