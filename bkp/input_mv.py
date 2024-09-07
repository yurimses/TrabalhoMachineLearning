import os
import pandas as pd

def preencher_com_moda(df):
    # Substitui strings vazias por NaN
    df.replace('', pd.NA, inplace=True)
    
    # Itera sobre cada coluna para calcular e preencher a moda
    for column in df.columns:
        # Calcular a moda (o valor mais frequente) da coluna, ignorando NaN
        moda = df[column].mode().iloc[0]
        
        # Preenche valores ausentes com a moda
        df[column] = df[column].fillna(moda)  # Atualiza a coluna de forma explícita
    
    return df

def processarPartes(input_folder, output_folder):
    
    # Verifica se a pasta de saída existe, se não existir, cria
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Itera por todos os arquivos na pasta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, filename.replace('.csv', '_moda.csv'))
            try:
                # Lê o arquivo CSV
                df = pd.read_csv(input_csv, engine='python', encoding='ISO-8859-1', delimiter=',')

                # Preenche valores em branco com a moda
                df_preenchido = preencher_com_moda(df)

                # Salva o DataFrame processado em um novo arquivo CSV
                df_preenchido.to_csv(output_csv, index=False)
                print(f"Processado e salvo: {output_csv}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

# Pasta de entrada com arquivos CSV divididos
input_folder = '/home/yuri/Desktop/test/CSV Agegrado'

# Pasta de saída para os arquivos processados
output_folder = '/home/yuri/Desktop/test/CSV MV'
processarPartes(input_folder, output_folder)
