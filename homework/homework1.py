
import pandas as pd
import numpy as np


# Question 1

print(pd.__version__)


# Question 2

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"

df = pd.read_csv(url)

display(df)


# Question 3

print(df.columns[df.isna().any()])


# Question 4

print(df.ocean_proximity.value_counts(dropna= False).value_counts().value_counts())


# Question 5

print(df.median_house_value[df.ocean_proximity == 'NEAR BAY'].mean())


# Question 6

print("df.total_bedrooms.mean() ==", df.total_bedrooms.mean(), '\n')
df.total_bedrooms.isna().fillna(df.total_bedrooms.mean()), '\n'
print("df.total_bedrooms.mean() ==", df.total_bedrooms.mean(), '\n')


# Question 7

print(df.ocean_proximity.value_counts())
a = df[df['ocean_proximity'] == 'ISLAND'][['housing_median_age', 'total_rooms', 'total_bedrooms']]
display(a)

X = a.to_numpy()
XTX = np.dot(X.T, X)
XTX_inv = np.linalg.inv(XTX)
y = np.array([950, 1300, 800, 1000, 1300])
w = np.dot(np.dot(XTX_inv, X.T), y)
last_element_of_w = w[-1]
print(last_element_of_w)

