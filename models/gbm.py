from lightgbm import LGBMRegressor


class GBM:
    def __init__(self, **params):
        self.params = dict(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            verbosity=-1,
            n_jobs=2,
            random_state=42,
            deterministic=True,
            force_col_wise=True,
        )
        self.params.update(params)

    def fit(self, X, y):
        self.model = LGBMRegressor(**self.params).fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
