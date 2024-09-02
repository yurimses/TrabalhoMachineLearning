import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

# Função para carregar os arquivos CSV e combinar em um único DataFrame
def carregar_dados(caminhos_csv):
    dfs = []
    for caminho in caminhos_csv:
        df = pd.read_csv(caminho)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Caminho da pasta onde estão os arquivos CSV
folder_path = '/home/yuri/Desktop/test/CSV MV'

# Listar todos os arquivos CSV na pasta
csv_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')])

# Selecionar 8 arquivos para treinamento e 2 para teste
train_files = csv_files[:8]
test_files = csv_files[8:10]

# Carregar os dados de treinamento e teste
df_train = carregar_dados(train_files)
df_test = carregar_dados(test_files)

# Seleção das features e o target
features = ['CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'PUERPERA',
            'CARDIOPATI', 'SIND_DOWN', 'HEPATICA', 'NEUROLOGIC', 'PNEUMOPATI',
            'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'OUT_MORBI']

target = 'CLASSI_FIN'

# joga as features e o target para varaiveis X e y
X_train = df_train[features].values
y_train = df_train[target].values

# Define X e y para teste
X_test = df_test[features].values
y_test = df_test[target].values

# Modelo árvore de decisão
model_dt = DecisionTreeClassifier(random_state=27)

# modelo arvore de decisao
model_dt.fit(X_train, y_train)

# faz a prediça~o usando conjunto de teste
y_pred = model_dt.predict(X_test)

# métrica acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
