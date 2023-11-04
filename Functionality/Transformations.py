from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import string

class Remove_id(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Remove the id column"""
        self.remove_column = "id"
        return self

    def transform(self, X):
        if self.remove_column in X.columns:
            X = X.drop(columns=self.remove_column)
        return X.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return [f"Reduce_{i}" for i in input_features][1:]

class Reduce_text_change(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        """Remove the id column"""
        # X is the column here
        return self

    def transform(self,X):
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

        temp = pd.concat([X,pd.Series(self.final)],axis = 1).to_numpy()[:,[-1]]
        print(temp.shape)
        return temp

    def get_feature_names_out(self, input_features=None):
        return [f"text_changed_{i}" for i in input_features]

class Reduce_activity(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        """Remove the id column"""
        return self

    def transform(self,X):
        # X is the column here
        # Find and replace values that start with "Move" in the 'activity' column
        self.final = []
        for i in X["activity"]:
            if i.startswith("Move"):
                self.final.append("Move")
            else:
                self.final.append(i)
        temp = pd.concat([X,pd.Series(self.final)],axis = 1).to_numpy()[:,[-1]]
        return temp

    def get_feature_names_out(self, input_features=None):
        return [f"activity_changed_{i}" for i in input_features]


class Reduce_event(BaseEstimator, TransformerMixin):

    def __init__(self):
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
        return [f"activity_changed_{i}" for i in self.features]


if __name__ == "__main__":

    Remove_id()



