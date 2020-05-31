import normalization as norm
import networkBuilding as nBuilding
import predict as predict
import drawGraph as draw
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
import operator
class Quipus:
    knn = None
    X_train = []
    Y_train = []
        
    def predict(self, X_test, Y_test=[]):
        result = []
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test)
        g, nbrs = nBuilding.networkBuildKnn(
            self.X_train, self.Y_train, self.knn, self.eRadius)
        results=[]
        for index, instance in enumerate(X_test):
            if(len(Y_test)==0):
                nBuilding.insertNode(g,nbrs,instance)
            else:    
                nBuilding.insertNode(g,nbrs,instance,Y_test[index])
            tmpResults = predict.prediction(g,g.graph["lnNet"]+index,self.deepNeighbors)
            results.append(tmpResults)
            maxIndex=np.argmax(tmpResults)
            result.append(g.graph["classNames"][maxIndex])
        # print(np.array(result))
        # print(np.array(Y_test))
        # print("Prediction Accuracy = {0}%".format(round(100 * float(np.mean(np.array(result) == np.array(Y_test))), 2)))
        return result

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.partitionOptimization = 0.5

    def __init__(self, knn=3, eRadius=0.5, deepNeighbors=2):
        self.knn = knn
        self.eRadius = eRadius
        self.deepNeighbors = deepNeighbors

    def get_params(self, deep=False):
        return {'knn': self.knn, 'eRadius': self.eRadius, 'deepNeighbors': self.deepNeighbors}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def getDataCSV(url, className="Class"):
    dataset = {}
    data = pd.read_csv(url, keep_default_na=False,  na_values=np.nan)
    if(len(data.values[0]) == 1):
        data = pd.read_csv(url, ";", keep_default_na=False,  na_values=np.nan)
    dataset['target'] = data[className].values
    dataset['data'] = data.drop(className, axis=1).values
    return dataset

dataset = getDataCSV("./dataset/wine.csv")
# X_train, X_predict, Y_train, Y_predict = train_test_split(
#     dataset['data'], dataset["target"], test_size=0.25)
# (X_train, X_predict) = norm.preprocess(X_train, X_predict)


# quipusClass=Quipus(knn=14,eRadius=0.5,deepNeighbors=3)
# quipusClass.fit(X_train,Y_train)
# quipusClass.predict(X_predict,[])



test=5
total=[]
for i in range(test):
    quipusClass=Quipus(knn=14,eRadius=0.5,deepNeighbors=5)
    kfold = KFold(n_splits=10, random_state=None, shuffle=True)
    scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy",cv=kfold)
    total.append(scores)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
total=np.array(total)
print("----\n->Accuracy Total: %0.2f (+/- %0.2f)" % (total.mean(), total.std() * 2))