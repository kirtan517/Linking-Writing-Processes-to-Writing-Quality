from sklearn.linear_model import SGDRegressor,LinearRegression
from sklearn.ensemble import RandomForestRegressor

def getRandomForestRegression():
    rfreg = RandomForestRegressor()
    params_grid = [
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
        }
    ]
    return rfreg, params_grid


def getLinearRegression():
    linalg = LinearRegression()
    params = [
        {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "n_jobs": [-1, 1, 2],
            "positive": [True, False],
            "copy_X": [True, False]
        }
    ]
    return linalg, params

def getSgdRegerssion():
    sgdreg = SGDRegressor()
    params_grid = [
        {
            "loss": ["squared_error"],
            "penalty": ["l2"]
        }
    ]
    return sgdreg,params_grid

def getModel(model_name):
    model,params = None,None
    if model_name == "LinearRegression":
        model,params = getLinearRegression()
    elif model_name == "SGDRegression":
        model,params = getSgdRegerssion()
    elif model_name == "RandomForestRegressor":
        model,params = getRandomForestRegression()
    return model,params