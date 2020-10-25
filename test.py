#%%
from quipus import HLNB_BC, Quipus, Quipus2
import tools
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GridSearchCV
import time

strNameDataset = "covid19"
dataset = tools.getDataFromCSV("./dataset/" + strNameDataset + ".csv")
print("DATASET: " + strNameDataset)
print("---------")
seed = int(time.time())
kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
print("Partitions: " + str(seed))

# grid_param = {
#     "knn": range(1, 10),
#     "ePercentile": [0.0, 0.5, 1.0],
#     "bnn": [1, 3, 5],
#     "alpha": [0.0, 0.5, 1.0],
# }
grid_param = {
    "knn": range(2),
    "ePercentile": [0.0],
    "bnn": [1],
    "alpha": [0.0],
}

quipusClass = Quipus2()

gd_sr = GridSearchCV(
    estimator=quipusClass, param_grid=grid_param, scoring="accuracy", cv=kfold
)
gd_sr.fit(dataset["data"], dataset["target"])

best_parameters = gd_sr.best_params_
print(best_parameters)
print(gd_sr.best_score_)
print("---------")

test = 10
total = []
knnTest = best_parameters["knn"]
ePercentile = best_parameters["ePercentile"]
bnnTest = best_parameters["bnn"]
alpha = best_parameters["alpha"]
print("knn: ", knnTest)
print("e-percentile: ", ePercentile)
print("bnn: ", bnnTest)
print("alpha: ", alpha)

for i in range(test):
    quipusClass = Quipus2(
        knn=knnTest, ePercentile=ePercentile, bnn=bnnTest, alpha=alpha
    )
    kfold = KFold(n_splits=10, random_state=i + seed, shuffle=True)
    scores = cross_val_score(
        quipusClass, dataset["data"], dataset["target"], scoring="accuracy", cv=kfold
    )
    total.append(scores)
total = np.array(total)
np.savetxt(
    "./Tests/"
    + str(seed)
    + " "
    + strNameDataset
    + " scores "
    + str(total.mean())
    + "pm"
    + str(total.std() * 2)
    + " "
    + str(quipusClass)
    + "knn "
    + str(knnTest)
    + " e "
    + str(ePercentile)
    + " bnn "
    + str(bnnTest)
    + " alpha "
    + str(alpha),
    total,
    delimiter=",",
)
print("--------")
print(total.mean(), total.std() * 2)
print("---------")

