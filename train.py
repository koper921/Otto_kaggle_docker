from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, log_loss
from sklearn.model_selection import train_test_split
from model import XGBoostClassifierWithEarlyStop
import pandas as pd


df_train = pd.read_csv(r"C:\Users\MyPC\Documents\AIC-partage\TC1\Projet kaggle\train.csv", index_col=0)
print("Training set has {0[0]} rows and {0[1]} columns".format(df_train.shape))


X = df_train.drop('target', 1).values
y = df_train['target'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pca = PCA(n_components=70)
pipeline = Pipeline([('pca', pca), ('xgb', XGBoostClassifierWithEarlyStop())])
pipeline.fit(x_train, y_train) # doctest: +ELLIPSIS

y_pred = pipeline.predict(x_test)
y_pred_proba = pipeline.predict_proba(x_test)


print("precision : " ,precision_score(y_test, y_pred, average='macro'))
print("log loss : ", log_loss(y_test, y_pred_proba))
