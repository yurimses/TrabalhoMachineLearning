# LOGISTIC REGRESSION


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

'''
referências para o código:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://www.kaggle.com/code/satishgunjal/multiclass-logistic-regression-using-sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
'''

file_path = '/home/yuri/Desktop/Teste com MV-MODA/CSV Brasil/dados_agregadosBrasilY.csv'            # DADOS COM MISSING VALUES BRASIL
# file_path = '/home/yuri/Desktop/Teste com MV-MODA/CSV_2009_2019/dados_agregados_2009_2019.csv'      # DADOS COM MISSING VALUES BRASIL 2009 - 2019
# file_path = '/home/yuri/Desktop/Teste com MV-MODA/CSV_2020_2024/dados_agregados_2020_2024.csv'      # DADOS COM MISSING VALUES BRASIL 2020 - 2024


# file_path = '/home/yuri/Desktop/Teste sem MV/CSV Brasil/dados_agregadosBrasilN.csv'                 # DADOS SEM MISSING VALUES
# file_path = '/home/yuri/Desktop/Teste sem MV/CSV_2009_2019/dados_agregados_2009_2019.csv'           # DADOS SEM MISSING VALUES BRASIL 2009 - 2019
# file_path = '/home/yuri/Desktop/Teste sem MV/CSV_2020_2024/dados_agregados_2020_2024.csv'           # DADOS SEM MISSING VALUES BRASIL 2020 - 2024

df = pd.read_csv(file_path)

# Seleção das features e do target
features = [
    'CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'CARDIOPATI', 'PNEUMOPATI', 
    'IMUNODEPRE', 'RENAL', 'OUT_MORBI', 'EVOLUCAO', 'CLASSI_FIN'
]

target = 'CLASSI_FIN'

# Atribui as features e o target para variáveis X e y
X = df[features].values
y = df[target].values

# divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# Criação do modelo de Regressão Logística para múltiplas classes
model_lr = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000, random_state=27)) # segunda referencia


# treina o modelo
model_lr.fit(X_train, y_train)

# faz a prediça~o usando conjunto de teste
y_pred = model_lr.predict(X_test)

# métrica acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
