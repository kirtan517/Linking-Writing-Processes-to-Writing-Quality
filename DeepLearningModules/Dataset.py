import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class WritingProcessDataset(Dataset):

    def __init__(self,vectorizer,df,target = "train"):
        self.vectorizer = vectorizer
        self.df = df
        self._target = target
        self._indexes = self.df["id"].unique()


    @property
    def target(self):
        return self._target

    @target.setter
    def target(self,target):
        self._target = target

    def __len__(self):
        """Return the length of the id's"""
        return len(self._indexes)

    def __getitem__(self, item):
        """Return the dictionary of the vectorized version of the index score value"""

        pass



if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join("..","Data","train_logs.csv"))
    # print(len(train_df["id"].unique())) # 2471



