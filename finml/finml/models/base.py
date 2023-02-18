import abc

class BaseModel(metaclass=abc.ABCMeta):
    """Modeling things"""

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> object:
        """Make predictions after modeling things"""

    def __call__(self, *args, **kwargs) -> object:
        """leverage Python syntactic sugar to make the models' behaviors like functions"""
        return self.predict(*args, **kwargs)
    
class Model(BaseModel):
    """Learnable Models"""

    def fit(self, X, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, feature) -> object:
        raise NotImplementedError()
    

    



