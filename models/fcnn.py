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

from typing import List, Literal, Optional
import torch.nn as nn
import torch
from models.base import BoilerPlate


class FCNNv1(nn.Module, BoilerPlate):
    def __init__(self,
                 input_dim: int,
                 num_classes: Optional[int] = 2,
                 task: Literal['classification',
                               'regression'] = "classification",
                 HIDDEN_DIMS: List = [128, 64, 32],
                 DROPOUT_PROB: float = 0.2,
                 ):
        # Beware that you need to initialize the nn.Module first
        nn.Module.__init__(self)

        BoilerPlate.__init__(self,
                             num_classes=num_classes,
                             task=task)

        layers = [nn.Linear(input_dim, HIDDEN_DIMS[0]),
                  nn.BatchNorm1d(HIDDEN_DIMS[0]),
                  nn.Dropout(DROPOUT_PROB), nn.ReLU()]

        for i in range(1, len(HIDDEN_DIMS)):
            layers += [nn.Linear(HIDDEN_DIMS[i-1], HIDDEN_DIMS[i]),
                       nn.BatchNorm1d(HIDDEN_DIMS[i]),
                       nn.Dropout(DROPOUT_PROB), nn.ReLU()]

        if task == "regression":
            layers.append(nn.Linear(HIDDEN_DIMS[-1], 1))

        elif task == "classification":
            if num_classes > 2:
                layers.append(nn.Linear(HIDDEN_DIMS[-1], num_classes))
            else:
                layers.append(nn.Linear(HIDDEN_DIMS[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.model(X)
