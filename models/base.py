from typing import Literal, Tuple, Type, TypeGuard
import torch.nn as nn
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator


class PytorchEstimator(BaseEstimator):
    def __init__(self,
                 model_class: Type[nn.Module],
                 model_arch: dict = None,
                 task: Literal['classification', 'regression'] = "classification",  # noqa E501
                 epochs: int = 50,
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 verbose: bool = False):
        if not isinstance(model_class, type):
            raise ValueError("Expected `model_class` to be a class, "
                             "not an instance. Pass the class name "
                             "(e.g., `FCNNv1`) instead of an "
                             "instantiated object.")

        self.model_class = model_class
        self.model_arch = model_arch or {}
        self.task = task
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
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
            validation_data: Tuple[NDArray, NDArray] = None):
        X, y = self._cast_torch(X, y, self.device)
        if validation_data is not None:
            X_val, y_val = self._cast_torch(*validation_data, self.device)
        num_classes = len(torch.unique(y)) if self.task == "classification" else None
        self.model = self.model_class(input_dim=X.shape[1],
                                      num_classes=num_classes,
                                      task=self.task,
                                      **self.model_arch).to(self.device)
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

            if validation_data is not None:
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
        return self

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
        if self.task == "regression":
            raise ValueError("`predict_proba` is not applicable "
                             " for regression tasks.")
        return self._get_predictions(X, return_probabilities=True)

    def fit_predict(self, X, y, validation_data=None):
        return self.fit(X, y, validation_data=validation_data).predict(X)     


class BoilerPlate:
    def __init__(self, num_classes, task):
        self.num_classes = num_classes
        self.task = task
        if task == "regression":
            self.loss_fn = nn.MSELoss()
        elif task == "classification":
            self.loss_fn = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()  # noqa E501

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if (self.task == "classification" and self.num_classes == 2) or self.task=="regression":
            return self.loss_fn(y_pred[:, 0], y_true.float())
        return self.loss_fn(y_pred, y_true.to(torch.int64))

    def predict(self, y_pred: torch.Tensor):
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
