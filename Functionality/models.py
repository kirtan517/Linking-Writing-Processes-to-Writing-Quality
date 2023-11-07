from sklearn.linear_model import SGDRegressor,LinearRegression
from sklearn.ensemble import RandomForestRegressor

def getRandomForestRegression(optuna,trial = None):
    if optuna == False:
        rfreg = RandomForestRegressor()
        params_grid = [
            {
                "n_estimators": [100,200,250,300],
                "max_depth": [10,20,30],
                "min_samples_split": [50,100,200],
                "min_samples_leaf": [50,100,300],
                "max_features": ["sqrt"],
                "bootstrap": [True],
            }
        ]
        return rfreg, params_grid
    else:
        rfreg = RandomForestRegressor()
        params_grid = [
            {
                "n_estimators": trial.suggest_int("n_estimators", 50, 350),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 50, 300),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 300),
                "max_features": ["sqrt"],
                "bootstrap": [True],
            }
        ]
        return rfreg, params_grid





def getLinearRegression(optuna,trial = None):
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

def getSgdRegerssion(optuna,trial = None):
    sgdreg = SGDRegressor()
    params_grid = [
        {
            "loss": ["squared_error"],
            "penalty": ["l2"]
        }
    ]
    return sgdreg,params_grid

def getModel(model_name,optuna = False,trial = None):
    model,params = None,None
    if model_name == "LinearRegression":
        model,params = getLinearRegression(optuna,trial)
    elif model_name == "SGDRegression":
        model,params = getSgdRegerssion(optuna,trial)
    elif model_name == "RandomForestRegressor":
        model,params = getRandomForestRegression(optuna,trial)
    return model,params