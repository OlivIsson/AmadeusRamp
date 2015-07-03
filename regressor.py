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
        self.clf = RandomForestRegressor(n_estimators=501, max_depth=100, max_features=22,n_jobs=-1)
        #self.clf = Pipeline([('scaler', StandardScaler()),
        #                     ("RF", RandomForestRegressor(n_estimators=50, max_depth = 10))])
        self.clf1 = Pipeline([('scaler', StandardScaler()),
                                  ("PCA", PCA(n_components=22)),
                                  ("GB",GradientBoostingRegressor(n_estimators = 25, max_depth=100))])
        #self.clf1 = Pipeline([('scaler', StandardScaler()),
                                  #("GB",GradientBoostingRegressor(n_estimators = 500, max_depth=6))])
        #self.clf2 = Pipeline([('scaler', StandardScaler()),
        #                      ("LR",LinearRegression())])

    def fit(self, X, y):
        self.clf.fit(X, y.reshape((y.shape[0],)))
        self.clf1.fit(X, y.reshape((y.shape[0],)))
        #self.clf2.fit(X, y.reshape((y.shape[0],)))
        

    def predict(self, X):
        #list_clf=[self.clf.predict(X),self.clf2.predict(X)]
        list_clf=[self.clf.predict(X),self.clf1.predict(X)]
        list_clf=[self.clf1.predict(X)]
        #return sum(list_clf)/float(len(list_clf))
        return self.clf.predict(X) * 0.87 + self.clf1.predict(X) * 0.13
		
		
