import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Tuple
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, validate_data


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

    @property
    def get_selected_wavenumbers(self) -> np.ndarray:
        self._check_is_fitted()
        return self.wavenumbers_[self.selected_indices]

    def _check_is_fitted(self):
        if not hasattr(self, "_is_fitted"):
            raise NotFittedError("This WavenumberWindowSelector1D "
                                 "is not fitted yet!")

class PDS(TransformerMixin):
    """
    Transfer model with Piecewise Direct standardization based on PLS.
    PLS can be replaced with Ridge/LR/PCR
    Ref: https://www.sciencedirect.com/science/article/pii/0169743995000747
    """
    def __init__(self, n_components_max: int=2, window_size: int = 5):
        self.n_components_max = n_components_max
        self.window_size = window_size

    def fit(self, X_master, X_slave) -> None:
        X_master = validate_data(
            self, X_master, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        X_slave = validate_data(
            self, X_slave, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        num_wavelengths = X_master.shape[1]

        if X_master.shape != X_slave.shape:
            raise ValueError("The number of samples and number of wavelengths for "
                             "both arrays should be equal.")
        
        self.transfer_matrix = np.zeros((num_wavelengths, num_wavelengths))

        self.mean_master = X_master.mean(axis=0)
        self.mean_slave = X_slave.mean(axis=0)

        X_master_centered = X_master - self.mean_master
        X_slave_centered = X_slave - self.mean_slave

        for i in range(num_wavelengths):
            # we are predicting the wavelengths of master instrument (one by one)
            y = X_master_centered[:, i]
            # The X is going to be spectra from the slave instrument
            lb = np.max([0, i-self.window_size])
            ub = np.min([i+self.window_size, num_wavelengths-1])
            wv_range_selected = np.arange(lb, ub + 1)
            X = X_slave_centered[:, wv_range_selected]

            chosen_num_components = np.min([len(wv_range_selected), self.n_components_max])

            pls = PLSRegression(chosen_num_components, scale=False).fit(X, y)
            # intercept is zero since it's already mean-centered.
            self.transfer_matrix[wv_range_selected, i] = pls.coef_
        return self

    def transform(self, X_slave) -> np.ndarray:
        """
        You can calculate the offset in the fit step and modify this step as follows:
        self.offset = self.mean_master - self.mean_slave@self.transfer_matrix
        In the transform step:
        X_slave@self.transfer_matrix + self.offset
        It's exactly the same if you expand:
        (X_test_slave - X_mean_slave)*W + X_master_mean
        X_test_slave*W + X_master_mean - X_mean_slave*W
        """
        check_is_fitted(self, "transfer_matrix")
        X_slaved_centered = X_slave - self.mean_slave
        return X_slaved_centered@self.transfer_matrix + self.mean_master

class PDS_TruncatedSVD(TransformerMixin):
    """
    Transfer model with Piecewise Direct standardization based on TruncatedSVD.
    """
    def __init__(self, n_components_max: int=2, window_size: int = 5):
        self.n_components_max = n_components_max
        self.window_size = window_size

    def fit(self, X_master, X_slave) -> None:
        X_master = validate_data(
            self, X_master, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        X_slave = validate_data(
            self, X_slave, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )

        num_wavelengths = X_master.shape[1]

        if X_master.shape != X_slave.shape:
            raise ValueError("The number of samples and number of wavelengths should be equal.")
        
        self.transfer_matrix = np.zeros((num_wavelengths, num_wavelengths))

        self.mean_master = X_master.mean(axis=0)
        self.mean_slave = X_slave.mean(axis=0)

        X_master_centered = X_master - self.mean_master
        X_slave_centered = X_slave - self.mean_slave

        for i in range(num_wavelengths):
            y = X_master_centered[:, i]
            lb = np.max([0, i-self.window_size])
            ub = np.min([i+self.window_size, num_wavelengths-1])
            wv_range_selected = np.arange(lb, ub + 1)
            X = X_slave_centered[:, wv_range_selected]

            chosen_num_components = np.min([len(wv_range_selected), self.n_components_max])
            U, S, Vt = np.linalg.svd(X, full_matrices=True)
            F = Vt.T[:, :chosen_num_components]@np.linalg.inv(np.diag(S[:3]))@U[:, :chosen_num_components].T@y
            self.transfer_matrix[wv_range_selected, i] = F
        return self

    def transform(self, X_slave) -> np.ndarray:
        check_is_fitted(self, "transfer_matrix")
        X_slaved_centered = X_slave - self.mean_slave
        return X_slaved_centered@self.transfer_matrix + self.mean_master
