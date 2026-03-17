import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

print("Ukupno je napravljeno",data.Model.count(), "mjerenja")
nondupes = data.drop_duplicates()
nondupes = nondupes.reset_index(drop=True)
print("Popis tipa podataka za svaku veličinu:")
print(data.dtypes)
print("Od",data.Model.count(),"vrijednosti,",data.Model.count()-nondupes.Model.count(),"je duplicirano")
nonempty = data.dropna(axis=0)
nonempty = nonempty.reset_index(drop=True)
print("Od",data.Model.count(),"vrijednosti,",nonempty.Model.count(),"je ispravno zapisano")
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].astype("category")
print("Pretvorba objekata u kategorije:")
print(data.dtypes)

print("Najveća 3 potrošitelja u gradu:")
print(data.nlargest(3,"Fuel Consumption City (L/100km)")[["Make","Model","Fuel Consumption City (L/100km)"]])
print("Najmanja 3 potrošitelja u gradu:")
print(data.nsmallest(3,"Fuel Consumption City (L/100km)")[["Make","Model","Fuel Consumption City (L/100km)"]])

velikiMotori = data[(data["Engine Size (L)"]>=2.5) & (data["Engine Size (L)"]<=3.5)]
print("Kolicina motora veca od 2.5L i manja od 3.5L:")
print(velikiMotori.Model.count())
print("Prosjecna emisija plinova za takva vozila:")
print(velikiMotori["CO2 Emissions (g/km)"].mean())

print("Izmjereno vozila marke Audi:")
print(data[(data["Make"]=="Audi")].Model.count())
print("Prosjecna emisija marke Audi s motorima od 4 cilindra:")
print(data[((data["Make"]=="Audi") & (data["Cylinders"]==4))]["CO2 Emissions (g/km)"].mean())

print("Prosjecna emisija vozila po kolicini cilindara:")
print(data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean())

#Z == Benzin
print(data[(data["Fuel Type"]=="Z")])