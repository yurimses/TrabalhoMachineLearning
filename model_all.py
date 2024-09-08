import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report

# Caminho do arquivo de dados
file_path = '/home/yuri/Desktop/Teste com MV-MODA/CSV Brasil/dados_agregadosBrasilY.csv'            # DADOS COM MISSING VALUES BRASIL
# file_path = '/home/yuri/Desktop/Teste com MV-MODA/CSV_2009_2019/dados_agregados_2009_2019.csv'      # DADOS COM MISSING VALUES BRASIL 2009 - 2019
# file_path = '/home/yuri/Desktop/Teste com MV-MODA/CSV_2020_2024/dados_agregados_2020_2024.csv'      # DADOS COM MISSING VALUES BRASIL 2020 - 2024


# file_path = '/home/yuri/Desktop/Teste sem MV/CSV Brasil/dados_agregadosBrasilN.csv'                 # DADOS SEM MISSING VALUES
# file_path = '/home/yuri/Desktop/Teste sem MV/CSV_2009_2019/dados_agregados_2009_2019.csv'           # DADOS SEM MISSING VALUES BRASIL 2009 - 2019
# file_path = '/home/yuri/Desktop/Teste sem MV/CSV_2020_2024/dados_agregados_2020_2024.csv'           # DADOS SEM MISSING VALUES BRASIL 2020 - 2024

# Carregar os dados
df = pd.read_csv(file_path)

# Seleção das features e o target
features = [
    'CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'CARDIOPATI', 'PNEUMOPATI', 'IMUNODEPRE', 'RENAL', 'OUT_MORBI', 'EVOLUCAO', 'CLASSI_FIN'
]
target = 'CLASSI_FIN'

# Preparar os dados
X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# Modelos
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=27),
    'Gaussian Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=27),
    'LinearSVC': LinearSVC(random_state=27),
    'Logistic Regression (OneVsRest)': OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000, random_state=27))
}

# Avaliar e mostrar resultados dos modelos
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f'Modelo: {model_name}')
    print(f'Acurácia: {accuracy * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'Precisão: {precision * 100:.2f}%')
    print(f'F1-Score: {f1 * 100:.2f}%')
    # print('Relatório de Classificação:')
    # print(classification_report(y_test, y_pred, target_names=df[target].unique().astype(str)))
    print('-' * 40)