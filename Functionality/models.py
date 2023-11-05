from sklearn.linear_model import SGDRegressor


def getLinearRegression():
    pass

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
    if model_name == "":
        model,params = getLinearRegression()
    elif model_name == "":
        model,params = getLinearRegression()
    return model,params