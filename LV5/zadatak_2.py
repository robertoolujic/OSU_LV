import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

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
y_train_unique = np.hstack([X_train, y_train])
print(y_train_unique)
plt.hist(y_train)
plt.hist(y_test)
plt.show()

#NOVO==================================================
# broj primjera po klasama (train)
classes_train, counts_train = np.unique(y_train, return_counts=True)

# broj primjera po klasama (test)
classes_test, counts_test = np.unique(y_test, return_counts=True)

# pretvori klase u nazive
labels_names = [labels[int(c)] for c in classes_train]

x = np.arange(len(classes_train))  # pozicije na x osi
width = 0.35

plt.figure(figsize=(8, 6))

# train stupci
plt.bar(x - width/2, counts_train, width, label='Train')

# test stupci
plt.bar(x + width/2, counts_test, width, label='Test')

plt.xticks(x, labels_names)
plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')
plt.title('Broj primjera po klasama (train vs test)')
plt.legend()
plt.grid(axis='y')


plt.show()

from sklearn.linear_model import LogisticRegression

# kreiranje modela (multiclass automatski)
model = LogisticRegression()

# učenje modela
model.fit(X_train, y_train)

# koeficijenti po klasi
print("Koeficijenti (theta1, theta2) po klasi:")
print(model.coef_)

# intercept po klasi
print("Intercept (theta0) po klasi:")
print(model.intercept_)

# crtanje decision regions za podatke za ucenje
plot_decision_regions(X_train, y_train, classifier=model)
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.title('Granice odluke logističke regresije (train)')
plt.show()

# predikcije
y_pred = model.predict(X_test)

# indeks dobro/pogrešno klasificiranih
correct_idx = (y_pred == y_test)
incorrect_idx = (y_pred != y_test)

plt.figure(figsize=(8,6))
plt.scatter(X_test[correct_idx, 0], X_test[correct_idx, 1], c='green', label='Točno klasificirani')
plt.scatter(X_test[incorrect_idx, 0], X_test[incorrect_idx, 1], c='black', label='Pogrešno klasificirani')
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.title('Testni skup - dobro/pogrešno klasificirani (originalni model)')
plt.legend()
plt.grid()
plt.show()

# ispis metrika
print("Točnost originalnog modela:", accuracy_score(y_test, y_pred))
print("Classification report originalnog modela:\n", classification_report(y_test, y_pred, target_names=['Adelie','Chinstrap','Gentoo']))

# ------------------------
# f) Dodavanje dodatnih ulaznih velicina i vizualizacija
# ------------------------
additional_inputs = ['bill_depth_mm', 'body_mass_g']  # dodatne značajke
input_variables_extended = input_variables + [col for col in additional_inputs if col in df.columns]

X_ext = df[input_variables_extended].to_numpy()
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_ext, y, test_size=0.2, random_state=123)

# učenje proširenog modela
model_ext = LogisticRegression(solver='lbfgs', max_iter=200)
model_ext.fit(X_train_ext, y_train_ext)

# predikcija
y_pred_ext = model_ext.predict(X_test_ext)

# indeks dobro/pogrešno klasificiranih
correct_idx_ext = (y_pred_ext == y_test_ext)
incorrect_idx_ext = (y_pred_ext != y_test_ext)

plt.figure(figsize=(8,6))
# 2D projekcija prvih dviju značajki
plt.scatter(X_test_ext[correct_idx_ext, 0], X_test_ext[correct_idx_ext, 1], c='green', label='Točno klasificirani')
plt.scatter(X_test_ext[incorrect_idx_ext, 0], X_test_ext[incorrect_idx_ext, 1], c='black', label='Pogrešno klasificirani')
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.title('Testni skup - dobro/pogrešno klasificirani (prošireni model)')
plt.legend()
plt.grid()
plt.show()

# ispis metrika proširenog modela
print("Točnost proširenog modela:", accuracy_score(y_test_ext, y_pred_ext))
print("Classification report proširenog modela:\n", classification_report(y_test_ext, y_pred_ext, target_names=['Adelie','Chinstrap','Gentoo']))
