class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
       

        # following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['quarter'] = X_encoded['DateOfDeparture'].dt.quarter
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        
        #fares = pd.read_csv("AirFares2012Q1to2013Q2.csv",sep=';')
        
        path = os.path.dirname(__file__)
                #data_weather = pd.read_csv(os.path.join(path, "data_weather.csv"))
        fares = pd.read_csv(os.path.join(path,"AirFares2012Q1to2013Q2.csv"), sep = ';')
                
        X_fares = fares [['ORIGIN', 'DEST', 'Quarter', 'TotalPax','TotalFare', 'AverageFare', 'Year']]
        X_encoded = X_encoded.merge(X_fares, how='left',
                    left_on=['Departure', 'Arrival','quarter','year'], 
                    right_on=['ORIGIN','DEST','Quarter','Year'], sort=False)
        
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        
        X_encoded = X_encoded.drop('ORIGIN', axis=1)
        X_encoded = X_encoded.drop('DEST', axis=1)
        X_encoded = X_encoded.drop('Quarter', axis=1)
        X_encoded = X_encoded.drop('Year', axis=1)
        X_encoded = X_encoded.drop('year', axis=1)
        X_encoded = X_encoded.drop('quarter', axis=1)
        X_encoded = X_encoded.drop('TotalFare', axis=1)
        X_encoded = X_encoded.drop('TotalPax', axis=1)
        
        aaa=X_encoded['AverageFare']
        ii=np.isnan(X_encoded['AverageFare'])
        rr=np.where(ii)
        jj=np.where(ii == True)[0]
        X_encoded['AverageFare'][jj]=0.0
        
        X_array = X_encoded.values
        return X_array
