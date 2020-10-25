#%%
from quipus import HLNB_BC, Quipus, Quipus2, Quipus3
import tools
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GridSearchCV
import time


# strNameDataset = "glcmCovid19"
strNameDataset = "covid19"
dataset = tools.getDataFromCSV("./dataset/" + strNameDataset + ".csv")
print("DATASET: " + strNameDataset)
print("---------")
seed = int(time.time())
quipusClass = Quipus3(knn=15)
quipusClass.fit(dataset["data"], dataset["target"])
# tools.drawGraphByClass(quipusClass.graphs[1])

# %%
tools.drawGraphByClass(quipusClass.graphs[0])
# tools.drawGraphByClass(quipusClass.graphs[1])
# tools.drawGraphByClass(quipusClass.graphs[2])
# tools.drawGraphByClass(quipusClass.graphs[3])
# tools.drawGraphByClass(quipusClass.graphs[5])

# %%

    # def __init__(self, knn=3, ePercentile=0.5, bnn=3, alpha=1.0):
    #     self.knn = knn
    #     self.bnn = bnn
    #     self.alpha = alpha
    #     self.ePercentile = ePercentile