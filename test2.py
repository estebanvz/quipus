#%%
from quipus import HLNB_BC, Quipus, Quipus2, Quipus3
import tools
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GridSearchCV
import time

for strNameDataset in ["teacher"]:
    dataset = tools.getDataFromCSV("./dataset/" + strNameDataset + ".csv")
    print("DATASET: " + strNameDataset)
    print("---------")
    seed = int(time.time())
    kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
    print("Partitions: " + str(seed))

    grid_param = {
        "knn": range(1,3),
        "ePercentile": [0.0,0.25,0.5],
        "bnn": [1],
        "alpha": [0.0,0.5,1.0],
    }

    quipusClass = Quipus3()

    gd_sr = GridSearchCV(
        estimator=quipusClass, param_grid=grid_param, scoring="accuracy", cv=kfold
    )
    gd_sr.fit(dataset["data"], dataset["target"])

    best_parameters = gd_sr.best_params_
    print(best_parameters)
    print(gd_sr.best_score_)
    print("---------")
    #%%

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
        quipusClass = Quipus3(
            knn=knnTest, ePercentile=ePercentile, bnn=bnnTest, alpha=alpha
        )
        kfold = KFold(n_splits=10, random_state=i + seed, shuffle=True)
        scores = cross_val_score(
            quipusClass,
            dataset["data"],
            dataset["target"],
            scoring="accuracy",
            cv=kfold,
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


# #%%
# asd = Quipus()
# dic = gd_sr.best_params_
# asd.set_params(knn=dic["knn"])


# # %%

# test = 20
# total = []
# knnTest = gd_sr.best_params_["knn"]
# eRadiusTest = 0.0
# bnnTest = 1
# alpha = 0.0
# print("knn: ", knnTest)
# print("e-radius: ", eRadiusTest)
# print("bnnTest: ", bnnTest)
# print("alpha: ", alpha)

# for i in range(test):
#     quipusClass = HLNB_BC(
#         knn=knnTest, ePercentile=eRadiusTest, bnn=bnnTest, alpha=alpha
#     )
#     kfold = KFold(n_splits=10, random_state=i + 42, shuffle=True)
#     scores = cross_val_score(
#         quipusClass, dataset["data"], dataset["target"], scoring="accuracy", cv=kfold
#     )
#     # scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy", cv=5)
#     total.append(scores)
#     asd = scores.tostring()
#     # print(scores)
#     # print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
# total = np.array(total)
# np.savetxt(strNameDataset + " scores HLNB_BC knn " + str(knnTest), total, delimiter=",")
# print("--------")
# print(total.mean(), total.std() * 2)
# print("---------")

# # print("----\n->Accuracy Total: %0.5f (+/- %0.5f)" % (total.mean(), total.std() * 2))
# # %%
# grid_param = {
#     "knn": range(1, 10),
#     "ePercentile": [0.0, 0.5],
#     "bnn": [1],
#     "alpha": [0.0],
# }
# quipusClass = Quipus()
# gd_sr = GridSearchCV(
#     estimator=quipusClass, param_grid=grid_param, scoring="accuracy", cv=10, n_jobs=-1
# )
# gd_sr.fit(dataset["data"], dataset["target"])
# best_parameters = gd_sr.best_params_
# print(best_parameters)
# print(gd_sr.best_score_)
# print("---------")

# #%%

# test = 20
# total = []
# knnTest = gd_sr.best_params_["knn"]
# eRadiusTest = 0.0
# bnnTest = 1
# alpha = 0.0
# print("knn: ", knnTest)
# print("e-radius: ", eRadiusTest)
# print("bnnTest: ", bnnTest)
# print("alpha: ", alpha)

# for i in range(test):
#     quipusClass = Quipus(knn=knnTest, ePercentile=eRadiusTest, bnn=bnnTest, alpha=alpha)
#     kfold = KFold(n_splits=10, random_state=i + 42, shuffle=True)
#     scores = cross_val_score(
#         quipusClass, dataset["data"], dataset["target"], scoring="accuracy", cv=kfold
#     )
#     # scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy", cv=5)
#     total.append(scores)
#     asd = scores.tostring()
#     # print(scores)
#     # print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
# total = np.array(total)
# np.savetxt(strNameDataset + " scores Quipus knn " + str(knnTest), total, delimiter=",")
# print("--------")
# print(total.mean(), total.std() * 2)
# print("----\n->Accuracy Total: %0.5f (+/- %0.5f)" % (total.mean(), total.std() * 2))
# print("--------")

# %%
