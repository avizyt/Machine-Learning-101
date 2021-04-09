import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]  # petal length , petal width
y = (iris['target'] == 2).astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=122)

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge')),
])
svm_clf.fit(X_train, y_train)

prediction = svm_clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print(accuracy)

# PLOTTING
plt.plot(y_test,prediction)
plt.xlabel('real value')
plt.ylabel('predicted value')
plt.title('Iris SVM Classification')
plt.show()

# if __name__ == '__main__':
#     val = 15
#     print("Actual value: ", y[val])
#     print("prediction value: ", svm_clf.predict(X)[val])