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

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf1 = RandomForestRegressor(n_estimators=501, max_depth=79, max_features=10)
 
    def fit(self, X, y):
        self.clf1.fit(X, y.reshape((y.shape[0],)))
 
    def predict(self, X):
        list_clf=[self.clf1.predict(X)]
        return sum(list_clf)/float(len(list_clf))
