import pandas as pd
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def getX_Y(train_logs_df,train_scores_df):
    final_df = pd.merge(train_logs_df,train_scores_df,on = "id",how = "inner")
    return train_logs_df,final_df["score"]

def perfromGridSearch(estimator,params,train_processed_df,train_logs_df,y,results = False,):
    """return : score, model"""
    grid_search = GridSearchCV(estimator=estimator,param_grid=params,scoring="neg_root_mean_squared_error",cv=6,return_train_score = True)
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

def performKfoldScore(model,train_processed_df,train_logs_df,y,k=5,optuna = False,trial = None):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    cv_scores = []


    # Perform k-fold cross-validation and store the results
    for i, (train_idx, test_idx) in enumerate(kf.split(train_processed_df)):
        X_train, X_test = train_processed_df.iloc[train_idx], train_processed_df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        temp_df = pd.concat([pd.Series(y), pd.Series(y_pred), train_logs_df["id"]], axis=1)
        final_df = temp_df.groupby("id").aggregate("mean")
        final_df.columns = ["y_true", "y_pred"]
        result = final_df[["y_true","y_pred"]].iloc[test_idx]
        score = mean_squared_error(result["y_true"], result["y_pred"])

        cv_scores.append(score)
        print(f"Fold {i + 1}: {score:.2f}")

    mean_score = sum(cv_scores) / len(cv_scores)
    std_dev = (sum((x - mean_score) ** 2 for x in cv_scores) / len(cv_scores)) ** 0.5
    print(f"Mean Score: {mean_score:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    if optuna:
        trial.set_user_attr('rmse', mean_score)
    return mean_score
