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
                return 0
            elif 15 <= idade <= 64:
                return 1
            elif idade >= 65:
                return 2
        except ValueError:
            pass
    return idade

def tratarSexo(sexo):
    if sexo == 'M':
        return 1
    elif sexo == 'F':
        return 2
    elif sexo == 'I':
        return 9
    return sexo

def tratarEscola(escola, ano):
    if ano in range(2009, 2019):
        if escola == '10':
            return 5
        elif escola == '2':
            return 3
        elif escola == '3':
            return 4
    return escola

def tratarClassi(classi, ano):
    if ano in range(2009, 2013):
        if classi == '2':
            return 3
        elif classi == '3':
            return 4
    return classi

def preencher_missing_values(df):
    for column in df.columns:
        moda = df[column].mode()[0]  # Calcula a moda da coluna
        df[column] = df[column].fillna(moda)  # Preenche os valores ausentes com a moda
    return df

def filtroFeaturesChunk(input_csv, output_csv, chunk_size=10000):
    features = [
        'CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'CS_ESCOL_N', 'CARDIOPATI', 'PNEUMOPATI', 
        'IMUNODEPRE', 'RENAL', 'OUT_MORBI', 'EVOLUCAO', 'CLASSI_FIN'
    ]
    
    first_chunk = True
    
    ano_str = os.path.basename(input_csv)[-6:-4] 
    ano = 2000 + int(ano_str)

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, engine='python', encoding='ISO-8859-1', delimiter=';'):
        chunk = chunk[chunk['EVOLUCAO'] == 1.0]  # Filtrar por não evoluiu a óbito
        
        # Adicionar a coluna do ano
        chunk['Ano'] = ano

        # Filtrar as colunas desejadas
        chunk = chunk[features + ['Ano']]
        
        chunk.replace('', np.nan, inplace=True)  # Substituir strings vazias por NaN
        
        # Aplicar o tratamento nas colunas
        chunk['NU_IDADE_N'] = chunk['NU_IDADE_N'].apply(tratarIdade)
        chunk['CS_SEXO'] = chunk['CS_SEXO'].apply(tratarSexo)
        
        if 'CS_ESCOL_N' in chunk.columns:
            chunk['CS_ESCOL_N'] = chunk['CS_ESCOL_N'].apply(lambda x: tratarEscola(x, ano))
        
        if 'CLASSI_FIN' in chunk.columns:
            chunk['CLASSI_FIN'] = chunk['CLASSI_FIN'].apply(lambda x: tratarClassi(x, ano))

        # Preencher valores ausentes com a moda de cada coluna
        chunk = preencher_missing_values(chunk)

        # Salvar o DataFrame filtrado em um arquivo CSV
        chunk.to_csv(output_csv, mode='a', header=first_chunk, index=False)
        first_chunk = False

        print(f"Processado chunk de tamanho {len(chunk)}")

    print(f"Arquivo filtrado salvo em {output_csv}")

def agregarArquivosChunk(output_folder, output_subfolder, chunk_size=10000):
    full_output_folder = os.path.join(output_folder, output_subfolder)
    if not os.path.exists(full_output_folder):
        os.makedirs(full_output_folder)

    dfs = []

    for filename in os.listdir(output_folder):
        if filename.endswith("_filtrado.csv"):
            file_path = os.path.join(output_folder, filename)
            try:
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, engine='python', encoding='ISO-8859-1', delimiter=','):
                    dfs.append(chunk)
                    print(f"Adicionado: {filename} - Chunk de tamanho {len(chunk)}")
            except Exception as e:
                print(f"Erro ao agregar {filename}: {e}")

    if dfs:
        df_aggregated = pd.concat(dfs, ignore_index=True)
        output_file = os.path.join(full_output_folder, 'dados_agregadosBrasilY.csv')
        df_aggregated.to_csv(output_file, index=False)
        print(f"Arquivo agregado salvo em {output_file}")
    else:
        print("Nenhum arquivo para agregar.")

def processarArquivosNaPasta(input_folder, output_folder, chunk_size=10000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, filename.replace('.csv', '_filtrado.csv'))
            try:
                filtroFeaturesChunk(input_csv, output_csv, chunk_size=chunk_size)
                print(f"Processado: {filename}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

# Pasta de entrada com CSV brutos
input_folder = '/home/yuri/Desktop/Datasetp'

# Pasta de saída com CSV tratados
output_folder = '/home/yuri/Desktop/Teste com MV-MODA'  

# Processar arquivos na pasta
processarArquivosNaPasta(input_folder, output_folder, chunk_size=10000)

# Agrupar todos os CSV em um só
output_subfolder = 'CSV Brasil'
agregarArquivosChunk(output_folder, output_subfolder, chunk_size=10000)
