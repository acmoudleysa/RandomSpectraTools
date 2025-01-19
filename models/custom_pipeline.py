from sklearn.base import TransformerMixin
import sklearn.pipeline
from sklearn.utils.metadata_routing import _routing_enabled
import sklearn
from sklearn.utils.validation import check_X_y


class CustomPipeline(sklearn.pipeline.Pipeline):
    def __init__(self, steps):
        super().__init__(steps)

    def fit(self, X, y, validation_data=None, **fit_params):
        if _routing_enabled():
            raise ValueError("This version of pipeline doesn't support metadata routing."
                             " Use sklearn.pipeline.Pipeline instead.")
        X, y = check_X_y(X, y, allow_nd=True)
        if validation_data is None:
            return super().fit(X, y, **fit_params)

        for name, step in self.steps[:-1]:
            if isinstance(step, TransformerMixin):
                print(f"Preprocessing training data with {name}")
                X = step.fit_transform(X, y)
            else:
                return TypeError(f"{name} is not a valid transformer")

        X_val, y_val = check_X_y(*validation_data, allow_nd=True)
        for name, step in self.steps[:-1]:
            X_val = step.transform(X_val)

        self.steps[-1][1].fit(X, y,
                              validation_data=(X_val, y_val),
                              **fit_params)
        self._is_fitted = True
        return self
    
    def predict_proba(self, X, **predict_proba_params):
        return super().predict_proba(X, **predict_proba_params)
    
    def predict(self, X, **predict_params):
        return super().predict(X, **predict_params)
    
    def fit_predict(self, X, y=None, **fit_params):
        return super().fit_predict(X, y, **fit_params)
    
    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted") and self._is_fitted
