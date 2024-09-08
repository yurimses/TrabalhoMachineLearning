import pandas as pd

# Carregar o arquivo CSV (substitua pelo caminho correto do arquivo)
arquivo_csv = '/home/yuri/Desktop/Teste com MV-MODA/CSV Brasil/dados_agregadosBrasilY.csv'
df = pd.read_csv(arquivo_csv)

# Para cada coluna, exibir os valores únicos
for coluna in df.columns:
    valores_unicos = df[coluna].unique()
    print(f"Coluna: {coluna}")
    print(f"Número de valores únicos: {len(valores_unicos)}")
    print(f"Valores únicos: {valores_unicos}")
    print("-" * 50)


# Ver a distribuição da coluna 'CLASSI_FIN' (substitua pelo nome da sua coluna de classes)
distribuicao_classes = df['CLASSI_FIN'].value_counts()

# Exibir a distribuição de classes para cada coluna
for coluna in df.columns:
    print(f"Distribuição das classes na coluna: {coluna}")
    distribuicao_classes = df[coluna].value_counts()
    print(distribuicao_classes)
    print("-" * 50)
