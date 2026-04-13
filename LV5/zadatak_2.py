import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()
# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

X_train_unique, X_train_unique_idx = np.unique(X_train, axis=0, return_index=True)
X_test_unique, X_test_unique_idx = np.unique(X_test, axis=0, return_index=True)

plt.hist(y_train, label="Trening podatci")
plt.hist(y_train[X_train_unique_idx], label="Jedinstveni trening podatci")
plt.hist(y_test, label="Test podatci")
plt.hist(y_test[X_test_unique_idx], label="Jedinstveni test podatci")
plt.legend(loc="lower center")
plt.show()
labels_inv= {'Adelie':0,'Chinstrap':1,'Gentoo':2}
y_train = np.array([labels_inv[val[0]] for val in y_train])
y_test  = np.array([labels_inv[val[0]] for val in y_test])
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train,y_train)

print("Parametri logističke regresije:")
print(LogRegression_model.coef_)
print("Intercept logističke regresije:")
print(LogRegression_model.intercept_)
#Svaka klasa ima svoje atribute, formira se 2x3 matrica

plot_decision_regions(X_train, y_train, classifier=LogRegression_model)
plt.show()
#Podatci su podijeljeni u 3 skupine, granica odluke ima neke od trening podataka već u krivim grupama

y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print("Matrica zabune: ", cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
print("Tocnost:", accuracy_score(y_test, y_test_p))
print(classification_report(y_test, y_test_p, target_names=["Adelie", "Chinstrap", "Gentoo"]))

extra_inputs = ["bill_depth_mm", "body_mass_g"]
input_variables = input_variables + [col for col in extra_inputs if col in df.columns]
print(input_variables)
X_new = df[input_variables].to_numpy()
y_new = df[output_variable].to_numpy()
X_train_new , X_test_new , y_train_new , y_test_new  = train_test_split(X_new , y_new , test_size = 0.2, random_state = 123)

y_train_new = np.array([labels_inv[val[0]] for val in y_train_new])
y_test_new  = np.array([labels_inv[val[0]] for val in y_test_new])
LogRegression_model_new = LogisticRegression()
LogRegression_model_new.fit(X_train_new,y_train_new)
print("Parametri logističke regresije:")
print(LogRegression_model_new.coef_)
print("Intercept logističke regresije:")
print(LogRegression_model_new.intercept_)
y_test_p_new = LogRegression_model_new.predict(X_test_new)
cm_new = confusion_matrix(y_test_new, y_test_p_new)
print("Matrica zabune: ", cm_new)
disp_new = ConfusionMatrixDisplay(cm_new)
disp_new.plot(cmap=plt.cm.Blues)
plt.show()

print("Tocnost:", accuracy_score(y_test_new, y_test_p_new))
print(classification_report(y_test_new, y_test_p_new, target_names=["Adelie", "Chinstrap", "Gentoo"]))
#Značajno poboljšanje u klasificiranju po gotovo svim metrikama
