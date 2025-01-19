"""
NOTE:
- Model architecture parameters (e.g., HIDDEN_DIMS, DROPOUT_PROB) should
be specified as uppercase.
    - Example:
        {'HIDDEN_DIMS': [128, 64], 'DROPOUT_PROB': 0.3}

- All custom model architectures should inherit both nn.Module and BoilerPlate.
    - Example:
        class FCNN(nn.Module, BoilerPlate)
    - Inside the constructor, nn.Module should be initialized
    before BoilerPlate exactly as given below (without modification).
        nn.Module.__init__(self)
        Boilerplate.__init__(self, num_classes=num_classes, task=task)
"""

from typing import Literal, Optional
import torch.nn as nn
import torch
from models.base import BoilerPlate


class SimpleCNN(nn.Module, BoilerPlate):
    def __init__(self,
                 input_dim: int,
                 num_classes: Optional[int] = 2,
                 task: Literal['classification',
                               'regression'] = "classification"
                 ):
        # Beware that you need to initialize the nn.Module first
        nn.Module.__init__(self)

        BoilerPlate.__init__(self,
                             num_classes=num_classes,
                             task=task)

        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(128, 28),
            nn.ReLU()
        )
        if task == "regression":
            self.model.append(nn.Linear(28, 1))

        elif task == "classification":
            if num_classes > 2:
                self.model.append(nn.Linear(28, num_classes))
            else:
                self.model.append(nn.Linear(28, 1))

    def forward(self, X: torch.Tensor):
        return self.model(X)

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.model:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)
