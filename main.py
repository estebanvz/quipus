#%%
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
from sklearn.model_selection import GridSearchCV
import operator
# https://github.com/ctgk/PRML/blob/master/notebooks/ch05_Neural_Networks.ipynb

class Quipus:
    knn = None
    X_train = []
    Y_train = []
    def score(self, X_test, Y_test):
         self.predict(X_test, Y_test)   
    def predict(self, X_test, Y_test=[]):
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test,1)
        result = []
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test)
        g, nbrs = nBuilding.networkBuildKnn(
            self.X_train, self.Y_train, self.knn, self.eRadius,labels=True)
        # nBuilding.getProperty(g)
        # draw.drawGraph(g,title="Graph Iris Dataset k="+str(self.knn)+" e="+str(self.eRadius)+ " b=10 α=0.0" )
        # draw.drawGraph(g,title="" )
        results=[]
        for index, instance in enumerate(X_test):
            indexNode=g.graph["lnNet"]+index
            if(len(Y_test)==0):
                nBuilding.insertNode(g,nbrs,instance)
            else:    
                nBuilding.insertNode(g,nbrs,instance,Y_test[index])
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.prediction(g,indexNode,self.deepNeighbors)
            results.append(tmpResults)
            maxIndex=np.argmax(tmpResults)
            newLabel=g.graph["classNames"][maxIndex]
            result.append(newLabel)
            # g.remove_node(str(indexNode))
            g.nodes[str(indexNode)]["label"]=newLabel
            nn = list(nx.neighbors(g,str(indexNode)))
            for node in nn:
                if(g.nodes[str(node)]["label"]!=newLabel):
                    g.remove_edge(str(node),str(indexNode))
            # draw.drawGraph(g,"Final Node")
            for edge in g.edges:
                g.edges[edge]["color"]="#9db4c0"
            g.graph["index"]+=1
        # draw.drawGraph(g,title="")
        
        if(len(Y_test)!=0):
            print("RESULT:",np.array(result))
            print("Y_TEST:",np.array(Y_test))
            acc=0
            err=[]
            err.append(g.graph["classNames"])
            for index,element in enumerate(result):
                if(element==Y_test[index]):
                    acc+=1
                else:
                    err.append([element,Y_test[index],results[index]])
            acc/=len(X_test)
            
            print("ERRORS: ", err)
            print("Accuracy ",round(acc,2),"%")
        return result
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.partitionOptimization = 0.5

    def __init__(self, knn=3, eRadius=0.5, deepNeighbors=1):
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

#%%
dataset = getDataCSV("./dataset/irisN.csv")
X_train, X_predict, Y_train, Y_predict = train_test_split(
    dataset['data'], dataset["target"], test_size=0.20)
# (X_train, X_predict) = norm.preprocess(X_train, X_predict,1)
#%%

from sklearn.datasets import make_classification
X, y = make_classification(n_informative=2,
                             n_clusters_per_class=1, n_classes=4)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax =plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

X_train, X_predict, Y_train, Y_predict = train_test_split(
    X, y, test_size=0.20)

quipusClass=Quipus(knn=5,eRadius=0.5,deepNeighbors=4)
quipusClass.fit(X_train,Y_train)
quipusClass.predict(X_predict,Y_predict)

#%%
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=3, n_features=5)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax =plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

X_train, X_predict, Y_train, Y_predict = train_test_split(
    X, y, test_size=0.20)

quipusClass=Quipus(knn=5,eRadius=0.5,deepNeighbors=4)
quipusClass.fit(X_train,Y_train)
quipusClass.predict(X_predict,Y_predict)
#%%
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.0)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax =plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

X_train, X_predict, Y_train, Y_predict = train_test_split(
    X, y, test_size=0.20)

quipusClass=Quipus(knn=5,eRadius=0.5,deepNeighbors=4)
quipusClass.fit(X_train,Y_train)
quipusClass.predict(X_predict,Y_predict)
#%%

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=100, noise=0.25)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax =plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

X_train, X_predict, Y_train, Y_predict = train_test_split(
    X, y, test_size=0.20)

quipusClass=Quipus(knn=3,eRadius=0.5,deepNeighbors=1)
quipusClass.fit(X_train,Y_train)
quipusClass.predict(X_predict,Y_predict)

#%%

# quipusClass=Quipus(knn=5,eRadius=0.5,deepNeighbors=4)
# quipusClass.fit(X_train,Y_train)
# quipusClass.predict(X_predict,Y_predict)


# []
# IRIS knn:7 deep 1
#IRIS 14/16
# wine knn:19 deep 1
# red Wine:3 deep 1 0.574
# f=open("results.txt",'w')
# f.close()

# grid_values = {'knn':range(1,20),'eRadius':[0.0,0.25,0.5]}
# kfold = KFold(n_splits=10)
# uruClass=Quipus()
# clf = GridSearchCV(uruClass, param_grid = grid_values,cv=kfold,scoring = 'accuracy',n_jobs=7)
# grid_result=clf.fit(dataset['data'],dataset['target'])
# print("Best Estimator: ",grid_result.best_estimator_.get_params(),' Score: ',grid_result.best_score_)

# f=open("results.txt",'a')
# f.write("Best Estimator: "+str(grid_result.best_estimator_.get_params())+' Score: '+str(grid_result.best_score_)+'\n')
# f.close()

# test=3
# total=[]
# knnTest=5
# eRadiusTest=0.5
# print ("knn: ",knnTest)
# print ("e-radius: ",eRadiusTest)
# for indexi, i in enumerate(range(test)):
#     quipusClass=Quipus(knn=knnTest,eRadius=eRadiusTest,deepNeighbors=1)
#     kfold = KFold(n_splits=5, random_state=indexi, shuffle=True)
#     scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy",cv=kfold)
#     # scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy", cv=5)
    
#     total.append(scores)
#     print(scores)
#     print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
# total=np.array(total)
# print("----\n->Accuracy Total: %0.4f (+/- %0.4f)" % (total.mean(), total.std() * 2))

 # %%
