import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Tuple
from sklearn.exceptions import NotFittedError


class WavenumberWindowSelector2D(BaseEstimator, TransformerMixin):
    """
    Make sure the data is EX*EM. Meaning a single 
    row is emission data for a particular excitation wavelength.
    """
    def __init__(self,
                 columns_range: Dict[str, List[int]],
                 wavenumbers: Dict[str, List[int]]):
        self.columns_range = columns_range
        self.wavenumbers = wavenumbers
        self.selected_indices = None

    def fit(self, X, y=None):
        self.wavenumbers_ = {k: np.array(v) for
                             k, v in self.wavenumbers.items()}
        self.selected_indices = {
            "Excitation": self._get_selected_indices("Excitation"),
            "Emission": self._get_selected_indices("Emission")
        }
        self._is_fitted = True
        return self

    def _get_selected_indices(self, key: str) -> np.ndarray:
        range_start, range_end = self.columns_range[key]
        wavenumbers = self.wavenumbers_[key]
        start_idx = np.argmin(np.abs(wavenumbers - range_start))
        end_idx = np.argmin(np.abs(wavenumbers - range_end)) + 1
        return np.arange(start_idx, end_idx, dtype=int)

    def transform(self, X):
        self._check_is_fitted()
        X = self._reshape_input(X)
        ex_indices = self.selected_indices["Excitation"]
        em_indices = self.selected_indices["Emission"]
        return X[:, ex_indices, :][:, :, em_indices].reshape(X.shape[0], -1)

    @property
    def get_selected_wavenumbers(self) -> Dict[str, np.ndarray]:
        self._check_is_fitted()
        return {key: self.wavenumbers_[key][indices]
                for key, indices in self.selected_indices.items()}

    def _check_is_fitted(self):
        if not hasattr(self, "_is_fitted"):
            raise NotFittedError("This WavenumberWindowSelector2D "
                                 "is not fitted yet!")

    def _reshape_input(self, X):
        n_ex, n_em = (len(self.wavenumbers_["Excitation"]),
                      len(self.wavenumbers_["Emission"]))
        if X.ndim == 2 and X.shape[1] == n_ex * n_em:
            return X.reshape(X.shape[0], n_ex, n_em)
        if X.ndim == 3 and X.shape[1:] == (n_ex, n_em):
            return X
        else:
            raise ValueError(f"Invalid shape {X.shape}."
                             f" Expected (n_samples, {n_ex} x {n_em}).")


class WavenumberWindowSelector1D(BaseEstimator, TransformerMixin):
    def __init__(self,
                 columns_range: List[float],
                 wavenumbers: List[float]) -> None:
        self.columns_range = columns_range
        self.wavenumbers = wavenumbers
        self.selected_indices = None

    def fit(self, X, y=None):
        self.wavenumbers_ = np.array(self.wavenumbers)
        self.selected_indices = np.arange(*self._get_selected_indices(),
                                          dtype=int)
        self._is_fitted = True
        return self

    def _get_selected_indices(self) -> Tuple[int, int]:
        start_idx = np.argmin(np.abs(self.wavenumbers_ - self.columns_range[0]))
        end_idx = np.argmin(np.abs(self.wavenumbers_ - self.columns_range[1])) + 1
        return start_idx, end_idx

    def transform(self, X):
        self._check_is_fitted()
        return X[:, self.selected_indices]

    def get_selected_wavenumbers(self) -> np.ndarray:
        self._check_is_fitted()
        return self.wavenumbers_[self.selected_indices]

    def _check_is_fitted(self):
        if not hasattr(self, "_is_fitted"):
            raise NotFittedError("This WavenumberWindowSelector1D "
                                 "is not fitted yet!")
