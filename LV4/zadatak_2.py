from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as mt
import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
data = pd.read_csv("data_C02_emission.csv")
X = data.select_dtypes(include="number").drop(columns="CO2 Emissions (g/km)")
X = X.reset_index (drop = True)
X["Fuel Type"] = data["Fuel Type"]
y = data["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =1 )

X_train_cat = ohe.fit_transform(X_train[["Fuel Type"]]).toarray()
X_test_cat = ohe.transform(X_test[["Fuel Type"]]).toarray()


X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

X_train_num = X_train.drop(columns="Fuel Type")
X_test_num = X_test.drop(columns="Fuel Type")

X_train = pd.DataFrame(np.hstack([X_train_num, X_train_cat]), columns = list(X_train_num.columns) + list(ohe.get_feature_names_out(["Fuel Type"])))
X_test = pd.DataFrame(np.hstack([X_test_num, X_test_cat]), columns = list(X_train_num.columns) + list(ohe.get_feature_names_out(["Fuel Type"])))

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

print("Parametri linearne regresije:")
print(linearModel.coef_)
print("Aproksimacijska funkcija je oblika:")
for i in range(0,linearModel.coef_.size):
    print("{} * x({})".format(linearModel.coef_[i],i), end="")
    if i<linearModel.coef_.size-1:
        print(" + ", end="")
    else:
        print("")

y_test_p = linearModel.predict(X_test)

abserror = abs(y_test_p-y_test)
y_test_series = pd.Series(y_test_p, index=y_test.index)
maxidx = abserror.idxmax()
print("Najveća pogreška:")
print(abserror.max())
print("vrijednost najveće pogreške:")
print("Stvarna vrijednost:", y_test.loc[maxidx])
print("Predikcija:", y_test_series.loc[maxidx])
print(data.loc[maxidx])
