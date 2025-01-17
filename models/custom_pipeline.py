from sklearn.base import BaseEstimator, clone, ClassifierMixin
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing
)
from sklearn.utils.metaestimators import available_if


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
