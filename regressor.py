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
        self.clf0 = RandomForestRegressor(n_estimators=300, max_depth=50, max_features=20)
        self.clf = GradientBoostingRegressor( n_estimators = 1500 , max_depth = 7 , max_features = 15)

    def fit(self, X, y):
        self.clf0.fit(X[:,0:102], y)
        self.clf.fit(X[:,102:], y)

    def predict(self, X):
        list_clf=[self.clf0.predict(X[:,0:102]),self.clf.predict(X[:,102:])]
        return sum(list_clf)/float(len(list_clf))
