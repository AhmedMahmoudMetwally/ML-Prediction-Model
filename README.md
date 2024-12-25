Data-Science-Methodology
#Data-Science-Methodology-

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import warnings
# Read data 

df=pd.read_csv(r'C:\Users\UP2store\Documents\bike_sales_100k.csv')
print (df)

#Checking missing values

print(df.isnull())
print(df.isnull().sum())
#Removing null 
df_cleaned = df.dropna()
df_cleaned
# drop duplicated 
df.drop_duplicates(inplace = True )
print(df.duplicated())


# label Encoding
from sklearn.preprocessing import LabelEncoder
for col in df.columns :
    le=LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(df)
