import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap="jet")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="jet", marker="x")
plt.show()

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train,y_train)

y_test_p = LogRegression_model.predict(X_test)
coef = LogRegression_model.coef_
print("Parametri linearne regresije:")
print(coef)
print("Aproksimacijska funkcija je oblika:")
for i in range(0,LogRegression_model.coef_.size):
    print("{} * x({})".format(LogRegression_model.coef_[:,i][0],i), end="")
    if i<LogRegression_model.coef_.size-1:
        print(" + ", end="")
    else:
        print("")
b=LogRegression_model.intercept_
w1,w2 = LogRegression_model.coef_.T

c=-b/w2
m=-w1/w2
x_bound = np.array([X.min(), X.max()])
y_bound = x_bound*m+c
plt.scatter(X_train[:,0], X_train[:,1], c=y_train,  cmap="jet")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="jet", marker="x")
plt.plot(x_bound,y_bound)
plt.show()

print("Tocnost:", accuracy_score(y_test, y_test_p))
print("Preciznost:", precision_score(y_test, y_test_p))
print("Odziv:", recall_score(y_test, y_test_p))
cm = confusion_matrix(y_test, y_test_p)
print("Matrica zabune:\n", cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

plt.scatter(X_test[:,0], X_test[:,1], c=np.where(y_test_p==y_test,"g","k"))
plt.plot(x_bound,y_bound)
plt.show()