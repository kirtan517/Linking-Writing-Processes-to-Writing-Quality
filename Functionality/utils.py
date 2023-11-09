import pandas as pd
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os

def getX_Y(train_logs_df,train_scores_df,perform_harmonic_variation = False,aggregation = False):
    if aggregation:
        return train_logs_df,train_scores_df["score"]
    final_df = pd.merge(train_logs_df,train_scores_df,on = "id",how = "inner")
    if perform_harmonic_variation:
        def HarmonicFunction(group):
            group["score"] = group["score"] * group["event_id"] / max(group["event_id"])
            return group

        final_df = final_df.groupby("id").apply(HarmonicFunction)
        final_df.reset_index(drop= True,inplace=True)
    return train_logs_df,final_df["score"]

def perfromGridSearch(estimator,params,train_processed_df,train_logs_df,y,results = False,aggregation = False):
    """return : score, model"""
    grid_search = GridSearchCV(estimator=estimator,param_grid=params,scoring="neg_root_mean_squared_error",cv=6,return_train_score = True)
    grid_search.fit(train_processed_df,y)
    best_model = grid_search.best_estimator_
    if aggregation:
        results = pd.DataFrame(grid_search.cv_results_)
        return best_model,results
    result = makePredictions(best_model,train_processed_df,y,train_logs_df)
    score = mean_squared_error(y,result["y_pred"])
    if results:
        results = pd.DataFrame(grid_search.cv_results_)
        return score,best_model,results
    else:
        return score,best_model

def makePredictions(model,X,train_logs_df,perform_harmonic_variation = False,aggregation = False):
    y_pred = model.predict(X)
    if aggregation:
        final_df = pd.DataFrame(pd.Series(y_pred), columns=["y_pred"])
        return final_df
    temp_df = pd.concat([pd.Series(y_pred),train_logs_df["id"]],axis = 1)
    final_df = temp_df.groupby("id").aggregate("mean")
    final_df.columns = ["y_pred"]
    return final_df

def aggreagateAlongId(train_processed_df,train_logs_df):
    temp_df = pd.concat([train_processed_df,train_logs_df["id"]],axis = 1)
    final_df = temp_df.groupby("id").agg("sum")
    final_df = final_df.reset_index()
    return final_df

def performCrossValidation(model,train_processed_df,train_logs_df,y,aggregation = False):
    """Return score"""
    # TODO : This is only reuturning train score also return test score
    results = cross_validate(model,train_processed_df,y,scoring="neg_root_mean_squared_error",cv =6,
                            return_train_score= True,return_estimator=True)
    if aggregation:
        results["test_score"] = results["test_score"] * -1
        results["train_score"] = results["train_score"] * -1
        results = pd.DataFrame(results)
        return results[["test_score","train_score"]]
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

def ConcatAlongId(train_processed_df,train_logs_df):
    try:
        train_processed_df.columns = [i.split("__")[1] for i in train_processed_df.columns]
    except:
        pass
    temp_df = pd.concat([train_processed_df,train_logs_df[["id","event_id"]]],axis = 1)
    return temp_df

if __name__ == "__main__":
    # For Testing purpose on the small dataset
    # Read the current files
    train_logs_directory = os.path.join("..", "Data", "train_logs.csv")
    train_scores_directory = os.path.join("..", "Data", "train_scores.csv")
    train_logs_df = pd.read_csv(train_logs_directory)
    train_scores_df = pd.read_csv(train_scores_directory)
    train_logs_df = train_logs_df.iloc[:100]
    train_scores_df = train_scores_df.iloc[:100]

    num_attributes = ["id", "event_id", "down_time", "up_time", "action_time", "cursor_position", "word_count"]
    final_df,y = getX_Y(train_logs_df,train_scores_df,perform_harmonic_variation= False,aggregation = True)
    print(y)
