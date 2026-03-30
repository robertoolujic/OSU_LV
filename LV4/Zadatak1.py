from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as mt
import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt

sc = MinMaxScaler()
data = pd.read_csv("data_C02_emission.csv")
X = data.select_dtypes(include="number").drop(columns="CO2 Emissions (g/km)")
X = X.reset_index (drop = True)
y = data["CO2 Emissions (g/km)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =1 )

X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

X_train_n = pd.DataFrame(X_train_n, columns=X_train.columns)
X_test_n = pd.DataFrame(X_test_n, columns=X_test.columns)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print("Parametri linearne regresije:")
print(linearModel.coef_)
print("Aproksimacijska funkcija je oblika:")
for i in range(0,linearModel.coef_.size):
    print("{} * x({})".format(linearModel.coef_[i],i), end="")
    if i<linearModel.coef_.size-1:
        print(" + ", end="")
    else:
        print("")

y_test_p = linearModel.predict(X_test_n)
MSE = mt.mean_squared_error(y_test, y_test_p)
RMSE = mt.root_mean_squared_error(y_test, y_test_p)
MAE = mt.mean_absolute_error(y_test, y_test_p)
MAPE = mt.mean_absolute_percentage_error(y_test, y_test_p)
print("Mean Square Error:")
print(MSE)
print("Root Mean Square Error:")
print(RMSE)
print("Mean Apsolute Error:")
print(MAE)
print("Mean Apsolute Percentage Error:")
print(MAPE)

plt.title("Dijagram raspršenja ovisnosti CO2 plinova o gradskoj potrošnji")
plt.scatter(X_train["Fuel Consumption City (L/100km)"], y_train, color="blue", marker=".")
plt.scatter(X_test["Fuel Consumption City (L/100km)"], y_test, color="red", marker=".")
plt.legend(["Skup za treniranje","Skup za testiranje"])
plt.show()

plt.subplot(1,2,1)
plt.title("Histogram gradske potrošnje prije skaliranja")
plt.hist(X_train["Fuel Consumption City (L/100km)"])
plt.subplot(1,2,2)
plt.title("Histogram gradske potrošnje poslije skaliranja")
plt.hist(X_train_n["Fuel Consumption City (L/100km)"])
plt.show()

plt.title("Dijagram raspršenja ovisnosti CO2 plinova o gradskoj potrošnji")
plt.scatter(X_test_n["Fuel Consumption City (L/100km)"], y_test_p, color="blue", marker=".")
plt.scatter(X_test_n["Fuel Consumption City (L/100km)"], y_test, color="red", marker=".")
plt.legend(["Predikcija modela","Stvarni rezultati"])
plt.show()

#Pri smanjenju ulaznih veličina (train skupa) povećava se greška, no najviše se povećava tek iznad 90%