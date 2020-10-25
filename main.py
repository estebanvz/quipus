#%%
from quipus import HLNB_BC, Quipus
import tools

# from main import Quipus,getDataCSV
# import normalization as norm
# import networkBuilding as nBuilding
# import predict as predict
import numpy as np

# import networkx as nx
# import matplotlib.pyplot as plt
# import math
# import pandas as pd
# from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GridSearchCV

# from sklearn.model_selection import GridSearchCV
strNameDataset='yeast'
dataset = tools.getDataFromCSV("./dataset/"+strNameDataset+".csv")
#%%
# X_train, X_predict, Y_train, Y_predict = train_test_split(
#     dataset['data'], dataset["target"], test_size=0.20)
# (X_train, X_predict) = norm.preprocess(X_train, X_predict,1)
#%%
# knnTest = 5
# ePercentile = 0.0
# bnnTest = 1
# quipusClass = Quipus(knn=knnTest, ePercentile=ePercentile, bnn=bnnTest, alpha=0.5)
# kfold = KFold(n_splits=5, random_state=42, shuffle=True)
# scores = cross_val_score(
#     quipusClass, dataset["data"], dataset["target"], scoring="accuracy", cv=kfold
# )
# print(scores)
# print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
#%%

# X_train, X_predict, Y_train, Y_predict = train_test_split(
#     dataset["data"], dataset["target"], test_size=0.25, stratify= dataset["target"]
# )
# knnTest = 1
# ePercentile = 0.0
# bnnTest = 1
# print('Quipus')
# quipusClass = Quipus(knn=knnTest, ePercentile=ePercentile, bnn=bnnTest, alpha=0.5)
# quipusClass.fit(X_train=X_train, Y_train=Y_train)
# quipusClass.predict(X_test=X_predict, Y_test=Y_predict)

# print('HLBNC')
# # bnnTest = 1
# # knnTest = 1
# quipusClass = HLNB_BC(knn=knnTest, ePercentile=ePercentile, bnn=bnnTest, alpha=0.5)
# quipusClass.fit(X_train=X_train, Y_train=Y_train)
# quipusClass.predict(X_test=X_predict, Y_test=Y_predict)
#%%
# tools.drawGraphs(quipusClass.graphs)

#%%
# k=np.array([1,2,3,4,5,6])
# asd=np.reshape(k,(-1,1))
#%%
# test=10
# total=[]
# knnTest=5
# eRadiusTest=0.0
# bnnTest=1
# print ("knn: ",knnTest)
# print ("e-radius: ",eRadiusTest)
# print ("bnnTest: ",bnnTest)
# for alphaTest in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#     print("alpha: ",alphaTest)
#     quipusClass=Quipus(knn=knnTest,eRadius=eRadiusTest,bnn=bnnTest,alpha=1.0)
#     kfold = KFold(n_splits=5, random_state=42, shuffle=True)
#     scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy",cv=kfold)
#     # scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy", cv=5)
#     total.append(scores)
#     print(scores)
#     print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
# total=np.array(total)
# print("----\n->Accuracy Total: %0.5f (+/- %0.5f)" % (total.mean(), total.std() * 2))

# np.savetxt("AlphaVariation"+str()+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),np.array(total),delimiter=",")
#%%
print("DATASET: "+strNameDataset)
print('---------')
# %%


grid_param = {"knn": range(1, 10), "ePercentile": [0.0,0.5], "bnn": [1], "alpha": [0.0]}
quipusClass = HLNB_BC()
gd_sr = GridSearchCV(
    estimator=quipusClass, param_grid=grid_param, scoring="accuracy", cv=10, n_jobs=-1
)
gd_sr.fit(dataset["data"], dataset["target"])
best_parameters = gd_sr.best_params_
print(best_parameters)
print(gd_sr.best_score_)
print('---------')



# %%

test = 20
total = []
knnTest = gd_sr.best_params_['knn']
eRadiusTest = 0.0
bnnTest = 1
alpha = 0.0
print("knn: ", knnTest)
print("e-radius: ", eRadiusTest)
print("bnnTest: ", bnnTest)
print("alpha: ", alpha)

for i in range(test):
    quipusClass = HLNB_BC(knn=knnTest, ePercentile=eRadiusTest, bnn=bnnTest, alpha=alpha)
    kfold = KFold(n_splits=10, random_state=i + 42, shuffle=True)
    scores = cross_val_score(
        quipusClass, dataset["data"], dataset["target"], scoring="accuracy", cv=kfold
    )
    # scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy", cv=5)
    total.append(scores)
    asd = scores.tostring()
    # print(scores)
    # print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
total = np.array(total)
np.savetxt(strNameDataset+" scores HLNB_BC knn "+str(knnTest), total, delimiter=",")
print("--------")
print(total.mean(), total.std() * 2)
print('---------')

# print("----\n->Accuracy Total: %0.5f (+/- %0.5f)" % (total.mean(), total.std() * 2))
# %%
grid_param = {"knn": range(1, 10), "ePercentile": [0.0,0.5], "bnn": [1], "alpha": [0.0]}
quipusClass = Quipus()
gd_sr = GridSearchCV(
    estimator=quipusClass, param_grid=grid_param, scoring="accuracy", cv=10, n_jobs=-1
)
gd_sr.fit(dataset["data"], dataset["target"])
best_parameters = gd_sr.best_params_
print(best_parameters)
print(gd_sr.best_score_)
print('---------')

#%%

test = 20
total = []
knnTest = gd_sr.best_params_['knn']
eRadiusTest = 0.0
bnnTest = 1
alpha = 0.0
print("knn: ", knnTest)
print("e-radius: ", eRadiusTest)
print("bnnTest: ", bnnTest)
print("alpha: ", alpha)

for i in range(test):
    quipusClass = Quipus(knn=knnTest, ePercentile=eRadiusTest, bnn=bnnTest, alpha=alpha)
    kfold = KFold(n_splits=10, random_state=i + 42, shuffle=True)
    scores = cross_val_score(
        quipusClass, dataset["data"], dataset["target"], scoring="accuracy", cv=kfold
    )
    # scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy", cv=5)
    total.append(scores)
    asd = scores.tostring()
    # print(scores)
    # print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
total = np.array(total)
np.savetxt(strNameDataset+" scores Quipus knn "+str(knnTest), total, delimiter=",")
print("--------")
print(total.mean(), total.std() * 2)
print("----\n->Accuracy Total: %0.5f (+/- %0.5f)" % (total.mean(), total.std() * 2))
print("--------")