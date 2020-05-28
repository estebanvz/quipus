from sklearn import preprocessing

def preprocess(train_data,test_data, normType=3):
    if(normType==1):
        scaler=preprocessing.StandardScaler().fit(train_data)
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    if(normType==2):
        scaler=preprocessing.MinMaxScaler().fit(train_data)
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    if(normType==3):
        scaler=preprocessing.Normalizer(norm='l2').fit(train_data)
        train_data=scaler.transform(train_data)
        test_data=scaler.transform(test_data)
    return train_data, test_data