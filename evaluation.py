#%%
import quipus
# from quipus import HLNB_BC, Quipus, Quipus2, Quipus3
import tools
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GridSearchCV
import imp  
imp.reload(tools)
imp.reload(quipus)

strNameDataset = "glcmCovid19"
dataset = tools.getDataFromCSV("./dataset/" + strNameDataset + ".csv")

print("DATASET: " + strNameDataset)
print("---------")
X_train, X_predict, Y_train, Y_predict = train_test_split(
    dataset["data"], dataset["target"], test_size=0.25, stratify=dataset["target"]
)
# (X_train, X_predict) = norm.preprocess(X_train, X_predict,1)
quipus = quipus.Quipus3(knn=5, ePercentile=0.0, bnn=3, alpha=0.0)
# print("Fit")
quipus.fit(X_train, Y_train)
# tools.drawGraphs(quipus.graphs)
quipus.predict(X_predict, Y_predict)
# print("Predict")
# tools.drawGraphs(quipus.graphs)
#%%
tools.drawGraphs(quipus.graphs)

# tools.drawGraphByClass(quipus.graphs[0])

#%%
grid_param = {
    "knn": range(1, 15),
    "ePercentile": [0.0, 0.5],
    "bnn": [1],
    "alpha": [0.0],
}
quipusClass = quipus.
gd_sr = GridSearchCV(
    estimator=quipusClass, param_grid=grid_param, scoring="accuracy", cv=10, n_jobs=-1
)
gd_sr.fit(dataset["data"], dataset["target"])
best_parameters = gd_sr.best_params_
print(best_parameters)
print(gd_sr.best_score_)
print("---------")


# %%

test = 20
total = []
knnTest = gd_sr.best_params_["knn"]
eRadiusTest = 0.0
bnnTest = 1
alpha = 0.0
print("knn: ", knnTest)
print("e-radius: ", eRadiusTest)
print("bnnTest: ", bnnTest)
print("alpha: ", alpha)

for i in range(test):
    quipusClass = HLNB_BC(
        knn=knnTest, ePercentile=eRadiusTest, bnn=bnnTest, alpha=alpha
    )
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
np.savetxt(strNameDataset + " scores HLNB_BC knn " + str(knnTest), total, delimiter=",")
print("--------")
print(total.mean(), total.std() * 2)
print("---------")

# print("----\n->Accuracy Total: %0.5f (+/- %0.5f)" % (total.mean(), total.std() * 2))
# %%
grid_param = {
    "knn": range(1, 10),
    "ePercentile": [0.0, 0.5],
    "bnn": [1],
    "alpha": [0.0],
}
quipusClass = Quipus()
gd_sr = GridSearchCV(
    estimator=quipusClass, param_grid=grid_param, scoring="accuracy", cv=10, n_jobs=-1
)
gd_sr.fit(dataset["data"], dataset["target"])
best_parameters = gd_sr.best_params_
print(best_parameters)
print(gd_sr.best_score_)
print("---------")

#%%

test = 20
total = []
knnTest = gd_sr.best_params_["knn"]
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
np.savetxt(strNameDataset + " scores Quipus knn " + str(knnTest), total, delimiter=",")
print("--------")
print(total.mean(), total.std() * 2)
print("----\n->Accuracy Total: %0.5f (+/- %0.5f)" % (total.mean(), total.std() * 2))
print("--------")

#%%
import matplotlib.pyplot as plt
plt.figure()
plt.boxplot(total.flatten())
# %%
