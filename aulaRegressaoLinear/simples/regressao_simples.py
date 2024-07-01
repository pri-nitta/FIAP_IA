import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dados = pd.read_excel(r"C:\Users\vntprni\Downloads\Sorvete.xlsx")
dados.head()

plt.scatter(dados['Temperatura'], dados['Vendas_Sorvetes'])
plt.xlabel('Temperatura (ºC)')
plt.ylabel('Vendas de sorvetes (milhares)')
plt.title('Relação entre temperatura e vendas de sorvetes')
plt.show()

dados.corr()

X = dados[['Temperatura']]
y = dados['Vendas_Sorvetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
previsoes = modelo.predict(X_test)

erro_medio_quadratico = mean_squared_error(y_test, previsoes)
erro_absoluto_medio = mean_absolute_error(y_test, previsoes)
r_quadrado = r2_score(y_test, previsoes)

print(f'erro médio quadrático: {erro_medio_quadratico}')
print(f'erro absoluto médio: {erro_absoluto_medio}')
print(f'R² (coeficiente de determinação): {r_quadrado}')

plt.scatter(X_test, y_test, label = 'real')
plt.scatter(X_test, previsoes, label='Previsto', color='red')
plt.xlabel('Temperatura ºC')