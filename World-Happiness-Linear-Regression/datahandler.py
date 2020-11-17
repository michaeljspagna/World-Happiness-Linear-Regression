from typing import Pattern
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
class DataHandler(object):
    """
    Loads, formats, and displays data
    """
    
    def __init__(self) -> None:
        self._loadData()
        self._createDataSets()
        
    
    def _loadData(self, ):
        path = 'happinessdata/'
        files = glob.glob(path + "*.csv")
        li = []
        for file in files:
            df = pd.read_csv(file, index_col=None, header=0)
            li.append(df)
        self.data = pd.concat(li, axis=0, ignore_index=True)

    def _createDataSets(self):
        self.displayData = self.data.drop(['Overall rank', 'Country or region'], axis=1)
        self.computeData = self.displayData.to_numpy()
        np.random.shuffle(self.computeData)
        self.computeData = np.insert(self.computeData, 1, 1, axis=1)
        self.computeData = np.round(self.computeData, decimals=3)
        
    def getTrainAndTest(self):
        cutoff = self.computeData.shape[0] // 4
        test = self.computeData[cutoff:]
        train = self.computeData[:cutoff]
        Y_test = test[:, 0]
        Y_test = np.reshape(Y_test, (Y_test.shape[0],1))
        X_test = test[:,1:]
        Y_train = train[:, 0]
        Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
        X_train = train[:, 1:]
        return X_test, Y_test, X_train, Y_train