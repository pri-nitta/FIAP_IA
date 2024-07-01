import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


dataset = pd.read_csv("housing.csv")
dataset.head()

np.random.seed(42)
#os para trabalhar com a base de dados

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

dataset.shape
dataset.info()

set(dataset["ocean_proximity"])
dataset['ocean_proximity'].value_counts()

#traz minimos e máximos, médias, desvio padrão...
dataset.describe()
dataset.hist(bins=50, figsize=(20,15))

df_train, df_test = train_test_split(dataset, test_size=0.2, random_state= 7)
print(len(df_train), "treinamento +", len(df_test), "teste")