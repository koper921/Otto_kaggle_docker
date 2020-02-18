from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# https://arxiv.org/ftp/arxiv/papers/1403/1403.1949.pdf

class XGBoostWithEarlyStop(BaseEstimator):
    def __init__(self, early_stopping_rounds=5, test_size=0.2,
                 eval_metric='mae', **estimator_params):
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        self.eval_metric = eval_metric = 'mlogloss'
        self.scaler = StandardScaler()
        self.sm = SMOTE(random_state=42)
        if self.estimator is not None:
            self.set_params(**estimator_params)

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def get_params(self, **params):
        return self.estimator.get_params()

    def fit(self, X, y):
        # fit scaling on all train
        self.scaler.fit(X)
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size)
        # smote resampling on the train dataset (not on validation set)
        x_train, y_train = self.sm.fit_resample(x_train, y_train)
        # scale echantillon
        x_train = self.scaler.transform(x_train)
        x_val = self.scaler.transform(x_val)

        # early stopping
        self.estimator.fit(x_train, y_train,
                           early_stopping_rounds=self.early_stopping_rounds,
                           eval_metric=self.eval_metric, eval_set=[(x_val, y_val)])
        return self

    def predict(self, X):

        return self.estimator.predict(self.scaler.transform(X))


class XGBoostClassifierWithEarlyStop(XGBoostWithEarlyStop):
    def __init__(self, *args, **kwargs):
        self.estimator = XGBClassifier()
        super(XGBoostClassifierWithEarlyStop, self).__init__(*args, **kwargs)



