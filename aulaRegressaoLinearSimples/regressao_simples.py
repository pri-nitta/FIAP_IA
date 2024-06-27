import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dados = pd.read_excel("Sorvete.xlsx")
dados.head()

plt.scatter(dados['Temperatura'], dados['Vendas_Sorvetes'])
plt.xlabel('Temperatura (ºC)')
plt.ylabel('Vendas de sorvetes (milhares)')
plt.title('Relação entre temperatura e vendas de sorvetes')
plt.show()

dados.corr()

