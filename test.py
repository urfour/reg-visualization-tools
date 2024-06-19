import pandas as pd

df = pd.read_csv('data/apartments_for_rent_classified_100K.csv', sep=';', index_col=0)
print(df.describe())
print(df.info())
print(df.head())