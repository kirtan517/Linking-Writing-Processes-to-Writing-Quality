from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import string
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import re


class Reduce_numerical_columns(BaseEstimator, TransformerMixin):
    def __init__(self,
                 isRemove_id=True,
                 isRemove_event_id=True,
                 isRemove_up_time=True,
                 isRemove_down_time=True,
                 add_difference_time=True):
        self.remove_id = None
        self.remove_event_id = None
        self.remove_up_time = None
        self.remove_down_time = None
        self.add_difference_time = add_difference_time
        self.isRemove_id = isRemove_id
        self.isRemove_event_id = isRemove_event_id
        self.isRemove_up_time = isRemove_up_time
        self.isRemove_down_time = isRemove_down_time

    def fit(self, X, y=None):
        """Remove the id column"""
        self.remove_id = "id"
        self.remove_event_id = "event_id"
        self.remove_up_time = "up_time"
        self.remove_down_time = "down_time"
        return self

    def transform(self, X):
        if self.add_difference_time:
            # Adding difference time absolute value
            X["difference_time"] = abs(X["down_time"] - X["up_time"])

        self.features = list(X.columns.to_numpy())

        # Do computation before removing the columns
        if self.remove_id in X.columns and self.isRemove_id:
            X = X.drop(columns=self.remove_id)
            self.features.remove(self.remove_id)
        if self.remove_event_id in X.columns and self.isRemove_event_id:
            X = X.drop(columns=self.remove_event_id)
            self.features.remove(self.remove_event_id)
        if self.remove_up_time in X.columns and self.isRemove_up_time:
            X = X.drop(columns=self.remove_up_time)
            self.features.remove(self.remove_up_time)
        if self.remove_down_time in X.columns and self.isRemove_down_time:
            X = X.drop(columns=self.remove_down_time)
            self.features.remove(self.remove_down_time)

        return X.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in self.features]


class Reduce_text_change(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.final = None

    def fit(self, X, y=None):
        """Remove the id column"""
        # X is the column here
        return self

    def transform(self, X):
        # TODO: Optimize this code with vertorize operations
        self.final = []
        for element in X["text_change"]:
            if "=>" in element:
                left, right = element.split("=>")
                self.final.append(len(right) - len(left))
            elif element == "NoChange":
                self.final.append(0)
            else:
                self.final.append(len(element))

        temp = pd.concat([X, pd.Series(self.final)], axis=1).to_numpy()[:, [-1]]
        self.features = ["text_change"]
        return temp

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in self.features]


class Reduce_activity(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.final = None

    def fit(self, X, y=None):
        """Remove the id column"""
        return self

    def transform(self, X):
        # X is the column here
        # Find and replace values that start with "Move" in the 'activity' column
        self.final = []
        for i in X["activity"]:
            if i.startswith("Move"):
                self.final.append("Move")
            else:
                self.final.append(i)

        temp = pd.concat([X, pd.Series(self.final)], axis=1).to_numpy()[:, [-1]]
        self.features = [""]
        return temp

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in input_features]


class Reduce_event(BaseEstimator, TransformerMixin):

    def __init__(self,name = "Up"):
        self.temp = None
        self.features = None
        self.storage = None
        self.name = name
        self.punchuations = ["'", '-', '!', '"', '#', '$', '%', '&', '(', ')', '*', ',', '.', '/', ':', ';', '?', '@',
                             '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '+', '<', '=', '>']
        self.characters = list(string.ascii_letters)
        self.operations = ['Alt', 'AltGraph', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'AudioVolumeDown',
                           'AudioVolumeMute', 'AudioVolumeUp', 'Backspace', 'Cancel', 'CapsLock', 'Clear',
                           'ContextMenu', 'Control', 'Dead', 'Delete', 'End', 'Enter', 'Escape', 'F1', 'F10', 'F11',
                           'F12', 'F15', 'F2', 'F3', 'F6', 'Tab', 'Space', 'Shift', 'ScrollLock', 'Rightclick',
                           'Process', 'Pause', 'PageUp', 'PageDown', 'OS', 'NumLock', 'ModeChange', 'Middleclick',
                           'Meta', 'MediaTrackPrevious', 'MediaTrackNext', 'MediaPlayPause', 'Leftclick', 'Insert',
                           'Home']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',',
                       'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.numbers = list(string.digits)

    def addRemaining(self,storage, key):
        for i in storage.keys():
            if i != key:
                storage[i].append(0)


    def manage(self, value):
        if value in self.punchuations:
            self.storage["Punchuations"].append(1.0)
            self.addRemaining(self.storage,"Punchuations")
        elif value in self.characters:
            self.storage["Characters"].append(1.0)
            self.addRemaining(self.storage,"Characters")
        elif value in self.numbers:
            self.storage["Numbers"].append(1.0)
            self.addRemaining(self.storage,"Numbers")
        elif value in self.operations:
            self.storage["Operations"].append(1.0)
            self.addRemaining(self.storage,"Operations")
        else:
            self.storage["Unknows"].append(1.0)
            self.addRemaining(self.storage,"Unknows")

    def manage2(self,value):
        if value in self.events:
            index = self.events.index(value)
            self.storage2[f"Events_{index}"].append(1.0)
            self.addRemaining(self.storage2,f"Events_{index}")
        else:
            index = self.events.index("Unidentified")
            self.storage2[f"Events_{index}"].append(1.0)
            self.addRemaining(self.storage2,f"Events_{index}")

    def fit(self, X, y=None):
        # Here X is the column
        return self

    def transform(self, X):
        name = X.columns[0]
        self.storage = {
            "Punchuations": [],
            "Characters": [],
            "Numbers": [],
            "Operations": [],
            "Unknows": [],
        }

        self.storage2 = {f"Events_{i}":[] for i,j in enumerate(self.events)}

        for i in X[name]:
            self.manage(i)
            self.manage2(i)

        self.temp = pd.DataFrame(self.storage)
        self.temp2 = pd.DataFrame(self.storage2)
        self.temp = pd.concat([self.temp,self.temp2],axis = 1)
        self.features = list(self.temp.columns)
        self.features = [self.name + "_" + s for s in self.features]

        return self.temp.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in self.features]


class Aggregation(BaseEstimator, TransformerMixin):
    def __init__(self, ):
        """Here assumption has been made that all the previous transformation is being added
        Also the id and event_id columns are been added to the dataset """
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.events =  ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',',
                       'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        aggregation_functions = {
            'action_time': ['sum', 'mean', 'std', 'min', 'max'],
            'cursor_position': ['sum', 'mean', 'std', 'min', 'max'],
            'word_count': ['sum','min','max'],
            'difference_time': ['sum', 'mean', 'std', 'min', 'max'],
            'text_change': ['sum', 'mean', 'std', 'min', 'max'],
            'activity_Input': ['sum'],
            'activity_Move': ['sum'],
            'activity_Nonproduction': ['sum'],
            'activity_Paste': ['sum'],
            'activity_Remove/Cut': ['sum'],
            'activity_Replace': ['sum'],
            # 'Up_Punchuations': ['sum'],
            # 'Up_Characters': ['sum'],
            # 'Up_Numbers': ['sum'],
            # 'Up_Operations': ['sum'],
            # 'Up_Unknows': ['sum'],
            'Down_Punchuations': ['sum'],
            'Down_Characters': ['sum'],
            'Down_Numbers': ['sum'],
            'Down_Operations': ['sum'],
            'Down_Unknows': ['sum'],
            'event_id': ['max'],
        }
        for i,j in enumerate(self.events):
            aggregation_functions[f"Down_Events_{i}"] = ["sum"]
        final_df = X.groupby("id").agg(aggregation_functions).reset_index()
        self.features = [f"{agg}_{col}" if agg != 'count' else col for col, agg in final_df.columns]
        return final_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in self.features]


if __name__ == "__main__":
    # For Testing purpose on the small dataset
    # Read the current files
    from utils import ConcatAlongId
    train_logs_directory = os.path.join("..", "Data", "train_logs.csv")
    train_scores_directory = os.path.join("..", "Data", "train_scores.csv")
    train_logs_df = pd.read_csv(train_logs_directory)
    train_scores_df = pd.read_csv(train_scores_directory)
    train_logs_df = train_logs_df.iloc[:5000]
    train_scores_df = train_scores_df.iloc[:5000]

    num_attributes = ["id", "event_id", "down_time", "up_time", "action_time", "cursor_position", "word_count"]

    processing = ColumnTransformer([
        ("RemoveId", make_pipeline(Reduce_numerical_columns()), num_attributes),
        ("ValueSum", make_pipeline(Reduce_text_change()), ["text_change", "id"]),
        ("RemoveMove", make_pipeline(Reduce_activity(), OneHotEncoder(sparse_output=False)), ["activity"]),
        # ("ReduceUpEvents",make_pipeline(Reduce_event(name = "Up")),["up_event"]),
        ("ReduceDownEvents", make_pipeline(Reduce_event(name="Down")), ["down_event"]),
    ],
        # remainder="passthrough"
    )

    train_processed_numpy = processing.fit_transform(train_logs_df)
    train_processed_df = pd.DataFrame(train_processed_numpy, columns=processing.get_feature_names_out())

    # Concating the columns
    train_postprocessed_df = ConcatAlongId(train_processed_df, train_logs_df)
    post_processing = make_pipeline(Aggregation())

    # Aggreagating the columns for both train and test
    train_postprocessed_numpy = post_processing.fit_transform(train_postprocessed_df)
    train_postprocessed_df = pd.DataFrame(train_postprocessed_numpy, columns=post_processing.get_feature_names_out())


