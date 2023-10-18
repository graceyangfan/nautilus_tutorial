from sklearn.preprocessing import MinMaxScaler,StandardScaler

class MinMaxNorm:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False 

    def fit(self,X):
        self.fitted = True 
        return self.scaler.fit(X)

    def fit_transform(self,X):
        self.fitted = True 
        return self.scaler.fit_transform(X)
         

    def transform(self,X):
        return self.scaler.transform(X)

    def inverse_transform(self,X):
        return self.scaler.inverse_transform(X) 

    def is_fitted(self):
        return self.fitted
    
class StandardNorm:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False 

    def fit(self,X):
        self.fitted = True 
        return self.scaler.fit(X)

    def fit_transform(self,X):
        self.fitted = True 
        return self.scaler.fit_transform(X)

    def transform(self,X):
        return self.scaler.transform(X)

    def inverse_transform(self,X):
        return self.scaler.inverse_transform(X) 

    def is_fitted(self):
        return self.fitted
    

