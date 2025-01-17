from typing import List, Literal, Optional, Tuple
import torch.nn as nn
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone, ClassifierMixin
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing
)
from matplotlib.pyplot import axes
from sklearn.utils.metaestimators import available_if


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List = [128, 64, 32],
                 num_classes: Optional[int] = 2,
                 task: Literal['classification',
                               'regression'] = "classification",
                 dropout_prob: float = 0.2):
        super().__init__()
        self.task = task
        self.num_classes = num_classes

        layers = [nn.Linear(input_dim, hidden_dims[0]),
                  nn.BatchNorm1d(hidden_dims[0]),
                  nn.Dropout(dropout_prob), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers += [nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                       nn.BatchNorm1d(hidden_dims[i]),
                       nn.Dropout(dropout_prob), nn.ReLU()]

        if task == "regression":
            layers.append(nn.Linear(hidden_dims[-1], 1))
        elif task == "classification":
            if num_classes > 2:
                layers.append(nn.Linear(hidden_dims[-1], num_classes))
            else:
                layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers)

        if task == "regression":
            self.loss_fn = nn.MSELoss()
        elif task == "classification":
            self.loss_fn = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()  # noqa E501

    def forward(self, X: torch.Tensor):
        return self.model(X)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Compute the loss based on the task.
        """
        if ((self.task == "classification" and self.num_classes == 2) or self.task=="regression"):
            return self.loss_fn(y_pred[:, 0], y_true.float())
        return self.loss_fn(y_pred, y_true.to(torch.int64))

    def predict(self, y_pred: torch.Tensor):
        """
        Convert raw model output to predictions based on the task.
       """
        if self.task == "classification":
            if self.num_classes == 2:
                return (torch.sigmoid(y_pred[:, 0]) >= 0.5).float()
            return torch.argmax(y_pred, dim=1)
        return y_pred.squeeze(1)

    def predict_proba(self, y_pred: torch.Tensor):
        """
        Convert raw model output to probabilities (classification only).
        """
        if self.task == "classification":
            if self.num_classes == 2:
                return torch.sigmoid(y_pred[:, 0])
            return torch.softmax(y_pred, dim=1)
        raise ValueError("Probabilities are not applicable for regression.")


class MLPEstimator(BaseEstimator):
    def __init__(self,
                 hidden_dims: List = [128, 64, 32],
                 task: Literal['classification', 'regression'] = "classification",  # noqa E501
                 epochs: int = 50,
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 dropout_prob: float = 0.2,
                 verbose: bool = False):
        self.hidden_dims = hidden_dims
        self.task = task
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob
        self.verbose = verbose
        self.device = torch.device('cuda'
                                   if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _cast_torch(X, y=None, device=None):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if y is not None:
            y_tensor = torch.tensor(y.flatten()).to(device)
            return X_tensor, y_tensor
        return X_tensor

    def fit(self,
            X: NDArray,
            y: NDArray,
            val_data: Tuple[NDArray, NDArray] = None):
        X, y = self._cast_torch(X, y, self.device)
        if val_data is not None:
            X_val, y_val = self._cast_torch(*val_data, self.device)
        num_classes = len(torch.unique(y)) if self.task == "classification" else None
        self.model = MLP(X.shape[1], self.hidden_dims,
                         num_classes, self.task,
                         self.dropout_prob).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        self.train_loss, self.validation_loss = [], []
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.model.compute_loss(y_pred, y)
            self.train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = self.model.compute_loss(val_pred, y_val)
                    self.validation_loss.append(val_loss.item())
                self.model.train()
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{self.epochs}], "
                          f"Training Loss: {loss.item():.4f}, "
                          f"Validation Loss: {val_loss.item():.4f}")

    def _get_predictions(self, X, return_probabilities=False):
        X = self._cast_torch(X, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
            if return_probabilities:
                return self.model.predict_proba(y_pred).cpu().numpy()
            return self.model.predict(y_pred).cpu().numpy()

    def predict(self, X):
        return self._get_predictions(X, return_probabilities=False)

    def predict_proba(self, X):
        return self._get_predictions(X, return_probabilities=True)


class CustomPipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, transformers, estimator):
        self.transformers = transformers
        self.estimator = estimator

    def get_metadata_routing(self):
        router = MetadataRouter(owner=self.__class__.__name__)
        router.add(
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
        )
        return router

    def fit(self, X, y, **fit_params):
        """Fit the pipeline."""
        routed_params = process_routing(self, "fit", **fit_params)
        val_data = routed_params.estimator.fit.get("val_data", None)

        # Sequentially fit and transform data through the transformers
        self.transformers_ = []
        if val_data is not None:
            X_val, y_val = val_data
        for transformer in self.transformers:
            transformer_ = clone(transformer).fit(X, y)
            X = transformer_.transform(X)
            if val_data is not None:
                X_val = transformer_.transform(X_val)
            self.transformers_.append(transformer_)

        self.estimator.fit(X, y, val_data=(X_val, y_val) if val_data else None)
        return self

    def predict(self, X):
        """Predict using the pipeline."""
        for transformer in self.transformers_:
            X = transformer.transform(X)
        return self.estimator.predict(X)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        """Predict probabilities if the estimator supports it."""
        for transformer in self.transformers_:
            X = transformer.transform(X)
        return self.estimator.predict_proba(X)
