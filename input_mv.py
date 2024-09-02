import os
import pandas as pd

def preencher_com_moda(df):
    # substiui strings vazias por NaN
    df.replace('', pd.NA, inplace=True)
    
    # itera sobre cada coluna para calcular e preencher a moda
    for column in df.columns:
        # calcular a moda (o valor mais frequente) da coluna, ignorando NaN
        moda = df[column].mode().iloc[0]
        
        # prencher valores ausentes com a moda
        df[column].fillna(moda, inplace=True)
    
    return df

def processarPartes(input_folder, output_folder):
    
    # verifica se a pasta de saída existe, se não existir, cria
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # itera por todos os arquivos na pasta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, filename.replace('.csv', '_moda.csv'))
            try:
                # lê o arquivo CSV
                df = pd.read_csv(input_csv, engine='python', encoding='ISO-8859-1', delimiter=',')

                # preenche valores em branco com a moda
                df_preenchido = preencher_com_moda(df)

                # salva o DataFrame processado em um novo arquivo CSV
                df_preenchido.to_csv(output_csv, index=False)
                print(f"Processado e salvo: {output_csv}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

# pasta de entrada com arquivos CSV divididos
input_folder = '/home/yuri/Desktop/test/CSV Agegrado'

# pasta de sadía para os arquivos processados
output_folder = '/home/yuri/Desktop/test/CSV MV'
processarPartes(input_folder, output_folder)
