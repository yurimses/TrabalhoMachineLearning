import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

'''
referências para o código:
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

'''

file_path = '/home/yuri/Desktop/test/CSV Agegrado/dados_agregados.csv'
df = pd.read_csv(file_path)

# seleçao dasffeatures e o target
features = ['CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'PUERPERA',
            'CARDIOPATI', 'SIND_DOWN', 'HEPATICA', 'NEUROLOGIC', 'PNEUMOPATI',
            'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'OUT_MORBI']

target = 'CLASSI_FIN'

# joga as features e o target para varaiveis X e y
X = df[features].values
y = df[target].values

# divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

# modelo arvore de decisao
model_dt = DecisionTreeClassifier(random_state=27)

# treina o modelo
model_dt.fit(X_train, y_train)

# faz a prediça~o usando conjunto de teste
y_pred = model_dt.predict(X_test)

# métrica acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
