#%%
import quipusNetwork as qn
import chaski
from sklearn.neighbors import NearestNeighbors
import quipus
import tools
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GridSearchCV
import imp

imp.reload(tools)
imp.reload(quipus)
imp.reload(qn)
#%%
strNameDataset = "glcmCovid19"
dataset = tools.getDataFromCSV("./dataset/" + strNameDataset + ".csv")


#%%
print("DATASET: " + strNameDataset)
print("---------")
X_train, X_predict, Y_train, Y_predict = train_test_split(
    dataset["data"], dataset["target"], test_size=0.25, stratify=dataset["target"]
)
#%%
g = qn.networkBuildKnn(
    X_train, Y_train, knn=5, ePercentile=None, subGraphsConnected=False
)
print("NODES: ", g.number_of_nodes())
print("EDGES: ", g.number_of_edges())

pathsGraphs = chaski.graphParticle(g)

nbrsByClass = []
for paths in pathsGraphs:
    measures = paths.getMeasures()
    nbrs = NearestNeighbors(n_neighbors=10, algorithm="ball_tree").fit(measures)
    nbrsByClass.append(nbrs)
prediction = []
for instance in X_predict:
    qn.insertNode(g, instance)

    pathsByClass = chaski.getInsertedChaskiPath(g)

    distancesByClass = []
    for index, paths in enumerate(pathsByClass):
        measures = paths.getMeasures()
        distances, indices = nbrsByClass[index].kneighbors(measures)
        distancesByClass.append(np.sum(distances) / len(measures))

    predicted = g.graph["classNames"][np.argmin(distancesByClass)]
    prediction.append(predicted)

print(np.array(prediction), Y_predict)
print("Accuracy ", (np.array(prediction) == Y_predict).sum() / len(Y_predict))
# for index, instance in enumerate(X_predict):
#     qn.insertNode(g, instance, label=Y_predict[index])
# tools.drawGraphByClass(g)
#%%
# (X_train, X_predict) = norm.preprocess(X_train, X_predict,1)
quipus = quipus.Quipus3(knn=12, ePercentile=0.1, bnn=1, alpha=0.0)
# print("Fit")
quipus.fit(X_train, Y_train)
# tools.drawGraphs(quipus.graphs)
quipus.predict(X_predict, Y_predict)
# print("Predict")
# tools.drawGraphs(quipus.graphs)# %%

# %%
