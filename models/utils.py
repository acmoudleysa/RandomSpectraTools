from matplotlib.pyplot import axes
from typing import List


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