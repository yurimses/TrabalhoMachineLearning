import os
import pandas as pd
import numpy as np

def tratarIdade(idade):

    if pd.notna(idade):
        try:
            idade_str = str(int(idade))  
            if len(idade_str) == 4:  
                idade = int(idade_str[-2:])

            if idade <= 14:
                # jovem
                return 0
            elif 15 <= idade <= 64:
                # adulto
                return 1
            elif idade >= 65:
                # idoso
                return 2
        except ValueError:
            pass

    return idade

def tratarSexo(sexo):
    if sexo == 'M':
        # masculino
        return 0  
    elif sexo == 'F':
        # feminino
        return 1
    else:
        return 2
    return sexo


def filtroFeatures(input_csv, output_csv):

    # features que queremos usar
    features = [
        'CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'PUERPERA',
        'CARDIOPATI', 'SIND_DOWN', 'HEPATICA', 'NEUROLOGIC', 'PNEUMOPATI', 
        'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'OUT_MORBI', 'EVOLUCAO'
    ]
    
    # ler o arquivo csv
    df = pd.read_csv(input_csv, engine='python', encoding='ISO-8859-1', delimiter=';')

    # filtrar as instâncias onde SG_UF_NOT é igual a 15 (PARÁ)
    # df_filtered = df[df['SG_UF_NOT'].isin([15, 11, 12, 13, 14, 16, 17])]

    df_filtered = df

    # filtrar por não evoluiu a óbito
    df_filtered = df_filtered[df_filtered['EVOLUCAO'] == 1.0]
    
    # filtrar as colunas desejadas no DataFrame já filtrado
    df_filtered = df_filtered[features]

    # substituir strings vazias por NaN
    df_filtered.replace('', np.nan, inplace=True)
    
    # remover linhas com valores ausentes ou NaN
    df_filtered = df_filtered.dropna()

    # aplicar o tratamento na coluna 'NU_IDADE_N'
    df_filtered['NU_IDADE_N'] = df_filtered['NU_IDADE_N'].apply(tratarIdade)
    df_filtered['CS_SEXO'] = df_filtered['CS_SEXO'].apply(tratarSexo)

     # Mostrar valores únicos de CS_SEXO e suas quantidades
    unique_sexo_values = df_filtered['CS_SEXO'].value_counts()
    print(f"Valores únicos e quantidades em CS_SEXO:\n{unique_sexo_values}")
    
    # salvar o novo DataFrame filtrado em um novo arquivo CSV
    df_filtered.to_csv(output_csv, index=False)
    
    return df_filtered

def processarArquivosNaPasta(input_folder, output_folder):

    # verifica se a pasta de saída existe, se não tiver, cria
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # verifica todos os arquivos .,csv
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, filename.replace('.csv', '_filtrado.csv'))
            try:
                # aplica a função de filtragem em cada arquivo
                filtroFeatures(input_csv, output_csv)
                print(f"Processado: {filename}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

def agregarArquivos(output_folder, output_subfolder):

    # cria a subpasta se não existir
    full_output_folder = os.path.join(output_folder, output_subfolder)
    if not os.path.exists(full_output_folder):
        os.makedirs(full_output_folder)

    # armazena os DataFrames
    dfs = []

    # verifica todos os arquivos CSV na pasta de saída
    for filename in os.listdir(output_folder):
        if filename.endswith("_filtrado.csv"):
            file_path = os.path.join(output_folder, filename)
            try:
                # lê cada arquivo CSV e adiciona o DataFrame à lista
                df = pd.read_csv(file_path, engine='python', encoding='ISO-8859-1', delimiter=',')

                # remover colunas vazias ou com todos os valores NA
                df = df.dropna(axis=1, how='all')

                # remover linhas com valores ausentes
                df = df.dropna()

                dfs.append(df)
                print(f"Adicionado: {filename}")
            except Exception as e:
                print(f"Erro ao agregar {filename}: {e}")

    # junta todos os DataFrames em um único DataFrame
    df_aggregated = pd.concat(dfs, ignore_index=True)

    # define o caminho para o arquivo agregado
    output_file = os.path.join(full_output_folder, 'dados_agregados.csv')
    
    # salva o DataFrame agregado em um novo arquivo CSV
    df_aggregated.to_csv(output_file, index=False)

    return df_aggregated

# pata de entrada com csv brutos
input_folder = '/home/yuri/Desktop/Datasetp'

# pasta de saída com cvs tratados
output_folder = '/home/yuri/Desktop/test'  
processarArquivosNaPasta(input_folder, output_folder)

# agrupar todos os csv em um só
output_subfolder = 'CSV Agegrado'
agregarArquivos(output_folder, output_subfolder)

