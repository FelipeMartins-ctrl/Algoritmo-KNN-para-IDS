import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv("/content/PDFMalware2022_pp.csv", dtype={"Class": int})

df.head()

df.describe()

df.info()

partA, partB = train_test_split(df, test_size=0.9)

  partA.info()
plt.hist(partA['Class'])
plt.xlabel('Labels')
plt.ylabel('Freq')
plt.show()

partB.info()
plt.hist(partB['Class'])
plt.xlabel('Labels')
plt.ylabel('Freq')
plt.show()

y = partA["Class"]
X = partA.drop("Class", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k_range = np.arange(1, 150, 2)
scores = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  score = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy").mean()
  scores.append(score)

  k_range = np.arange(1, 150, 2)
scores = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  score = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy").mean()
  scores.append(score)

  y = partB["Class"]
X = partB.drop("Class", axis=1)

clf1 = KNeighborsClassifier(n_neighbors = 3)
clf1_pred = cross_val_predict(clf1, X, y, cv=10)
conf_clf1 = confusion_matrix(y, clf1_pred)

clf1 = KNeighborsClassifier(n_neighbors = 3)
clf1_pred = cross_val_predict(clf1, X, y, cv=10)
conf_clf1 = confusion_matrix(y, clf1_pred)

clf1 = KNeighborsClassifier(n_neighbors = 7, weights = 'distance', algorithm = 'auto',metric = 'cityblock')
clf1_pred = cross_val_predict(clf1, X, y, cv=10)
conf_clf1 = confusion_matrix(y, clf1_pred)
tn = conf_clf1[0,0]
tp = conf_clf1[1,1]
fp = conf_clf1[0,1]
fn = conf_clf1[1,0]

print("TN:", tn)
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print()
print("Accuracy:", accuracy_score(y, clf1_pred)*100)
print("Precision:", precision_score(y, clf1_pred)*100)
print("Recall:", recall_score(y, clf1_pred)*100)

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report

knn_params = {
    "n_neighbors": range(1, 10, 2),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski", "cityblock"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "n_jobs": [-1],
    "p": [1, 2],
    "leaf_size": [15, 30, 45, 60]
}

knn = KNeighborsClassifier()

seed = 42

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
grid_search = GridSearchCV(estimator=knn, param_grid=knn_params, n_jobs=-1, cv=cv, scoring="accuracy")
grid_result = grid_search.fit(X_train, y_train)

best_model = knn.set_params(**grid_result.best_params_)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(grid_result.best_params_)

clf1 = KNeighborsClassifier(n_neighbors = 3, weights = 'distance', algorithm ='ball_tree', leaf_size = 15,  metric = 'manhattan', p = 2, n_jobs = -1)
clf1_pred = cross_val_predict(clf1, X, y, cv=10)
conf_clf1 = confusion_matrix(y, clf1_pred)
tn = conf_clf1[0,0]
tp = conf_clf1[1,1]
fp = conf_clf1[0,1]
fn = conf_clf1[1,0]

print("TN:", tn)
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print()
print("Accuracy:", accuracy_score(y, clf1_pred)*100)
print("Precision:", precision_score(y, clf1_pred)*100)
print("Recall:", recall_score(y, clf1_pred)*100)

![image](https://github.com/user-attachments/assets/1a66fdbc-d013-4e63-ad7a-6d42a3133598)
