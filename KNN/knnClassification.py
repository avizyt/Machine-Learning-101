import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
# print(data.head())

X = data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]
# print(X, y)

# converting the data

Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

# print(X)
#y
label_mapping = {
     'unacc':0,
     'acc':1,
     'good':2,
     'vgood':3

 }
y['class'] = y['class'].map(label_mapping)
y = np.array(y)
# print(y)

#create Model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

# test train splites

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)
# print("predictions: ", prediction)
# print("accuracy: ", accuracy)

if __name__ == "__main__":
    a = 1243
    print("Actual value: ", y[a])
    print("prediction value: ", knn.predict(X)[a])