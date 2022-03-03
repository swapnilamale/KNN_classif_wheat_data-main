# -*- coding: utf-8 -*-
# kNN classification
# multiclass classification
# dataset: wheat

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import neighbors
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import preprocessing


# read file
path = "F:/aegis/4 ml/dataset/supervised/classification/wheat/wheat.csv"
data = pd.read_csv(path)

data.head()
data.tail()
data.dtypes
data.shape
data.info()
data.describe()

# check the distribution of the y-variable
data.type.value_counts()

# standardize the data (only features have to be standardized)
# StandardScaler
# MinMaxScaler

# make a copy of the dataset
data_std = data.copy()

ss = preprocessing.StandardScaler()
sv = ss.fit_transform(data_std.iloc[:,:])
data_std.iloc[:,:] = sv

# restore the original Y-value in the data_std
data_std.type = data.type

# compare the actual and transformed data
data.head()
data_std.head()

# shuffle the dataset
data_std = data_std.sample(frac=1)
data_std.head(20)

# data_std.shape

# split the data into train/test
trainx,testx,trainy,testy=train_test_split(data_std.drop('type',1),
                                           data_std.type,
                                           test_size=0.2)
trainx.shape,trainy.shape
testx.shape,testy.shape

# cross-validation to determine the best K
cv_accuracy = []

n_list = np.arange(3,12,2); n_list

for n in n_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(model,trainx,trainy,cv=10,scoring='accuracy')
    cv_accuracy.append(scores.mean() )

print(cv_accuracy)  

bestK = n_list[cv_accuracy.index(max(cv_accuracy))]
print("best K = ", bestK)

# plot the Accuracy vs Neighbours to determine the best K
plt.plot(n_list,cv_accuracy)
plt.xlabel("Neighbours")
plt.ylabel("Accyuracy")
plt.title("Accuracy - Neighbours")

# build the model using the best K
m1 = neighbors.KNeighborsClassifier(n_neighbors=bestK).fit(trainx,trainy)
# metric = "manhattan"
# predict on test data
p1 = m1.predict(testx)

# confusion matrix and classification report
df1=pd.DataFrame({'actual':testy,'predicted':p1})

pd.crosstab(df1.actual,df1.predicted,margins=True)
print(classification_report(df1.actual,df1.predicted))

