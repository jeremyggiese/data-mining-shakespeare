
import pandas as pd

df = pd.read_csv("Shakespeare_clearer_data.csv")
#print(df.head())
#Separates each playerLine and places that result in a new dataset
newDF = df["Play"]+df["PlayerLine"].str.split(" ",expand=True)
#finalDF = pd.DataFrame(columns = ["Play", "Word"])
print(newDF.head())

#[Column][Row]
#print(finalDF)
#print(df)


print("done")

