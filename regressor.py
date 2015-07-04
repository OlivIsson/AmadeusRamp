from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.decomposition import KernelPCA
from sklearn import neighbors
from sklearn.decomposition import PCA
import xgboost as xgb

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = xgb.XGBRegressor(max_depth=30, n_estimators=1500, learning_rate=0.05)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
