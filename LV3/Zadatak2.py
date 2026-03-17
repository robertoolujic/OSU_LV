import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].astype("category")
data.plot.scatter(x="Fuel Consumption City (L/100km)",
                  y="CO2 Emissions (g/km)",
                  c="Fuel Type",
                  colormap="jet"
                  )
plt.show()
#Primjetljivo je kako svaki tip goriva ima jasno definiranu grupu podataka

grouped = data.groupby("Fuel Type")
grouped.boxplot(column=["Fuel Consumption Hwy (L/100km)"])

data.boxplot(column=["Fuel Consumption Hwy (L/100km)"])
plt.show()