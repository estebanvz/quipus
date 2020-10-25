#%%
from quipus import HLNB_BC, Quipus, Quipus2, Quipus3
import tools
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, GridSearchCV
import time
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


strNameDataset = "glcmCovid19"
dataset = tools.getDataFromCSV("./dataset/" + strNameDataset + ".csv")
print("DATASET: " + strNameDataset)
print("---------")
seed = int(time.time())




kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed+42)
print("Partitions: " + str(seed+42))

grid_param = {
    "n_neighbors": range(1, 20)
}
clsf = KNeighborsClassifier()
gd_sr = GridSearchCV(
    estimator=clsf,
    param_grid=grid_param,
    scoring="accuracy",
    cv=kfold.split(dataset["data"], dataset["target"]),
)
gd_sr.fit(dataset["data"], dataset["target"])

knn_parameters = gd_sr.best_params_
print(knn_parameters)
print(gd_sr.best_score_)
print("---------")

grid_param = {
    "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
}
clsf = LogisticRegression()
gd_sr = GridSearchCV(
    estimator=clsf,
    param_grid=grid_param,
    scoring="accuracy",
    cv=kfold.split(dataset["data"], dataset["target"]),
)
gd_sr.fit(dataset["data"], dataset["target"])

lr_parameters = gd_sr.best_params_
print(lr_parameters)
print(gd_sr.best_score_)
print("---------")

grid_param = {
    "solver": ['svd','lsqr','eigen'],
}
clsf = LinearDiscriminantAnalysis()
gd_sr = GridSearchCV(
    estimator=clsf,
    param_grid=grid_param,
    scoring="accuracy",
    cv=kfold.split(dataset["data"], dataset["target"]),
)
gd_sr.fit(dataset["data"], dataset["target"])

lda_parameters = gd_sr.best_params_
print(lda_parameters)
print(gd_sr.best_score_)
print("---------")

grid_param = {
    "criterion":['gini','entropy'],
    "splitter":['best','random'],
    "max_depth":[None,2,4,8,10]
}
clsf = DecisionTreeClassifier()
gd_sr = GridSearchCV(
    estimator=clsf,
    param_grid=grid_param,
    scoring="accuracy",
    cv=kfold.split(dataset["data"], dataset["target"]),
)
gd_sr.fit(dataset["data"], dataset["target"])

cart_parameters = gd_sr.best_params_
print(cart_parameters)
print(gd_sr.best_score_)
print("---------")


grid_param = {
    "kernel":['linear','poly','rbf','sigmoid'],
}
clsf = SVC()
gd_sr = GridSearchCV(
    estimator=clsf,
    param_grid=grid_param,
    scoring="accuracy",
    cv=kfold.split(dataset["data"], dataset["target"]),
)
gd_sr.fit(dataset["data"], dataset["target"])

svc_parameters = gd_sr.best_params_
print(svc_parameters)
print(gd_sr.best_score_)
print("---------")


grid_param = {
    "activation":['identity','logistic','tanh','relu'],
    "solver":['lbfgs','sgd','adam'],
    "max_iter": [200,400,600,1000]
}
clsf = MLPClassifier()
gd_sr = GridSearchCV(
    estimator=clsf,
    param_grid=grid_param,
    scoring="accuracy",
    cv=kfold.split(dataset["data"], dataset["target"]),
)
gd_sr.fit(dataset["data"], dataset["target"])

mlp_parameters = gd_sr.best_params_
print(mlp_parameters)
print(gd_sr.best_score_)
print("---------")




models = []
models.append(("LR", LogisticRegression().set_params(**lr_parameters)))
models.append(("LDA", LinearDiscriminantAnalysis().set_params(**lda_parameters)))
models.append(("KNN", KNeighborsClassifier().set_params(**knn_parameters)))
models.append(("CART", DecisionTreeClassifier().set_params(**cart_parameters)))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC().set_params(**svc_parameters)))
models.append(("MLP", MLPClassifier().set_params(**mlp_parameters)))


#%%

times=50
results = []
names = []
scoring = "accuracy"
for name, model in models:
    aux_results=[]
    names.append(name)
    for i in range(times):
        kfold = StratifiedKFold(n_splits=2, random_state=seed+i, shuffle=True)
        cv_results = cross_val_score(
            model,
            dataset["data"],
            dataset["target"],
            cv=kfold.split(dataset["data"], dataset["target"]),
            scoring=scoring,
        )
        aux_results.append(cv_results)
    finalScores=[e for subList in aux_results for e in subList]
    finalScores=np.array(finalScores)
    msg = "%s: %f (%f)" % (name, finalScores.mean(), finalScores.std())
    print(msg)
    results.append(finalScores)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()