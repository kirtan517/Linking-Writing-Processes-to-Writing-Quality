from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import string
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
        return temp

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in input_features]


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
        return temp

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in input_features]


class Reduce_event(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.temp = None
        self.features = None
        self.storage = None
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
        self.numbers = list(string.digits)

    def addRemaining(self, key):
        for i in self.storage.keys():
            if i != key:
                self.storage[i].append(0)

    def manage(self, value):
        if value in self.punchuations:
            self.storage["Punchuations"].append(1.0)
            self.addRemaining("Punchuations")
        elif value in self.characters:
            self.storage["Characters"].append(1.0)
            self.addRemaining("Characters")
        elif value in self.numbers:
            self.storage["Numbers"].append(1.0)
            self.addRemaining("Numbers")
        elif value in self.operations:
            self.storage["Operations"].append(1.0)
            self.addRemaining("Operations")
        else:
            self.storage["Unknows"].append(1.0)
            self.addRemaining("Unknows")

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
        for i in X[name]:
            self.manage(i)

        self.temp = pd.DataFrame(self.storage)
        self.features = list(self.temp.columns)

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
        temp_df = X.groupby("id")
        aggregation_functions = {
            'action_time': ['sum', 'mean', 'std', 'min', 'max'],
            'cursor_position': ['sum', 'mean', 'std', 'min', 'max'],
            'word_count': 'sum',
            'difference_time': ['sum', 'mean', 'std', 'min', 'max'],
            'text_change': ['sum', 'mean', 'std', 'min', 'max'],
            'activity_Input': ['sum'],
            'activity_Move': ['sum'],
            'activity_Nonproduction': ['sum'],
            'activity_Paste': ['sum'],
            'activity_Remove/Cut': ['sum'],
            'activity_Replace': ['sum'],
            'Punchuations': ['sum'],
            'Characters': ['sum'],
            'Numbers': ['sum'],
            'Operations': ['sum'],
            'Unknows': ['sum'],
            'event_id': ['max'],
        }
        final_df = temp_df.agg(aggregation_functions)
        self.features = [f"{agg}_{col}" if agg != 'count' else col for col, agg in final_df.columns]
        return final_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return [f"{i}" for i in self.features]


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

    processing = ColumnTransformer([
        ("RemoveId", make_pipeline(Reduce_numerical_columns(), StandardScaler()), num_attributes),
    ])

    train_processed_numpy = processing.fit_transform(train_logs_df)
    train_processed_df = pd.DataFrame(train_processed_numpy, columns=processing.get_feature_names_out())
