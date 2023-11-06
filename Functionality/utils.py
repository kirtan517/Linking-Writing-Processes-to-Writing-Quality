import pandas as pd
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.metrics import mean_squared_error

def getX_Y(train_logs_df,train_scores_df):
    final_df = pd.merge(train_logs_df,train_scores_df,on = "id",how = "inner")
    return train_logs_df,final_df["score"]

def perfromGridSearch(estimator,params,train_processed_df,train_logs_df,y,results = False,):
    """return : score, model"""
    grid_search = GridSearchCV(estimator=estimator,param_grid=params,scoring="neg_root_mean_squared_error",cv=3,return_train_score = True)
    grid_search.fit(train_processed_df,y)
    best_model = grid_search.best_estimator_
    result = makePredictions(best_model,train_processed_df,y,train_logs_df)
    score = mean_squared_error(result["y_true"],result["y_pred"])
    if(results):
        results = pd.DataFrame(grid_search.cv_results_)
        return score,best_model,results
    else:
        return score,best_model

def makePredictions(model,X,y_true,train_logs_df):
    y_pred = model.predict(X)
    temp_df = pd.concat([pd.Series(y_true),pd.Series(y_pred),train_logs_df["id"]],axis = 1)
    final_df = temp_df.groupby("id").aggregate("mean")
    final_df.columns = ["y_true","y_pred"]
    return final_df

def performCrossValidation(model,train_processed_df,train_logs_df,y):
    """Return score"""
    results = cross_validate(model,train_processed_df,y,scoring="neg_root_mean_squared_error",cv =6,
                            return_train_score= True,return_estimator=True)
    scores = []
    for i in results["estimator"]:
        result = makePredictions(i,train_processed_df,y,train_logs_df)
        score = mean_squared_error(result["y_true"],result["y_pred"])
        scores.append(score)
    return scores