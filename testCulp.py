from culp.classifier import culp
import numpy
import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
class CULPO:
    knn = None
    X_train = []
    Y_train = []
        
    def predict(self, X_test, Y_test=[]):
        lp='AA'
        # print("before ",self.X_train[:][:1])
        self.X_train, X_test = preprocess(self.X_train, X_test)
        # print("after ",self.X_train[:][:1])
        result=culp(self.X_train, self.Y_train, X_test, link_predictor=lp, similarity='manhattan', k=self.knn)
        return result
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.partitionOptimization = 0.5
    def __init__(self, knn=3):
        self.knn = knn

    def get_params(self, deep=False):
        return {'knn': self.knn}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def preprocess(train_data, test_data):
    '''preprocessing by normalization'''
    M = numpy.mean(train_data, axis=0)
    S = numpy.std(train_data, axis=0)
    S[S == 0] = M[S == 0] + 10e-10  # Controling devision by zero
    return (train_data - M) / S, (test_data - M) / S
wine = numpy.array(pandas.read_csv('culp/data/wine.txt', header=None))
data = wine[:, 1:].astype(float)
labels = wine[:, 0]
dataset={}
dataset['data']=data
dataset['target']=labels
test=10
total=[]
for i in range(test):
    quipusClass=CULPO(knn=12)
    kfold = KFold(n_splits=10, random_state=None, shuffle=True)
    scores = cross_val_score(quipusClass,dataset['data'],dataset['target'],scoring="accuracy",cv=kfold)
    total.append(scores)
    print(scores)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
total=numpy.array(total)
print("----\n->Accuracy Total: %0.4f (+/- %0.4f)" % (total.mean(), total.std() * 2))
