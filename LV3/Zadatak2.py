import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/robiy/Downloads/OSU_LV-master/OSU_LV-master/LV3/data_C02_emission.csv")
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].astype("category")
data["CO2 Emissions (g/km)"].plot(kind="hist", bins=20)
#Histogram prikazuje koliko vozila ima koju količinu CO2 emisija, i najviše vozila ima između 150 i 350 g/km emisija
plt.show()

data.plot.scatter(x="Fuel Consumption City (L/100km)",
                  y="CO2 Emissions (g/km)",
                  c="Fuel Type",
                  colormap="jet"
                  )
#Primjetljivo je kako svaki tip goriva ima jasno definiranu grupu podataka
plt.show()


data.boxplot(column=["Fuel Consumption Hwy (L/100km)"], by="Fuel Type")
plt.show()

agg = data.groupby("Fuel Type").agg({
    "Model": "count",
    "CO2 Emissions (g/km)": "mean"
})

agg.plot(kind="bar")
plt.show()