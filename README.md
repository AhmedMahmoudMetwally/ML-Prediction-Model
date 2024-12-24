# Data-Science-Methodology-
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
# Read data 
df=pd.read_csv(r'C:\Users\Pro\Downloads\Data Sience Methodology\?????????.csv')
print (df)
print(df.head())
# Checking missing values
print(df.isnull().sum())
print(df.isnull().sum().sum())
#Removing rows containing missing data
df_cleaned = df.dropna()
df
print(df.columns)
# Dropping insignificant columns
df_cleaned = df_cleaned.drop(columns=['Column_to_remove'], errors='ignore')
print(df_cleaned.info())
