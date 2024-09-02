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
        'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'OUT_MORBI', 'EVOLUCAO', 'CLASSI_FIN'
    ]
    
    # ler o arquivo csv
    df = pd.read_csv(input_csv, engine='python', encoding='ISO-8859-1', delimiter=';')

    # filtrar as instâncias onde SG_UF_NOT é igual a 15 (PARÁ)
    # df_filtered = df[df['SG_UF_NOT'].isin([15, 11, 12, 13, 14, 16, 17])]

    df_filtered = df

    # filtrar por não evoluiu a óbito
    df_filtered = df_filtered[df_filtered['EVOLUCAO'] == 1.0]

    # excluir instâncias onde OUT_MORBI é igual a 9
    # df_filtered = df_filtered[df_filtered['OUT_MORBI'] != 9]

    # excluir instâncias onde CLASSI_FIN é igual a 4
    # df_filtered = df_filtered[df_filtered['CLASSI_FIN'] != 4]
    
    # filtrar as colunas desejadas no DataFrame já filtrado
    df_filtered = df_filtered[features]

    # substituir strings vazias por NaN
    df_filtered.replace('', np.nan, inplace=True)
    
    # remover linhas com valores ausentes ou NaN
    # df_filtered = df_filtered.dropna()

    # aplicar o tratamento na coluna 'NU_IDADE_N'
    df_filtered['NU_IDADE_N'] = df_filtered['NU_IDADE_N'].apply(tratarIdade)
    df_filtered['CS_SEXO'] = df_filtered['CS_SEXO'].apply(tratarSexo)

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

                dfs.append(df)
                print(f"Adicionado: {filename}")
            except Exception as e:
                print(f"Erro ao agregar {filename}: {e}")

    # junta todos os DataFrames em um único DataFrame
    df_aggregated = pd.concat(dfs, ignore_index=True)

    # embaralhar as linhas
    df_aggregated = df_aggregated.sample(frac=1, random_state=27).reset_index(drop=True)

    # calcula o tamanho de cada parte
    part_size = len(df_aggregated) // 10

    # salvar em partes
    for i in range(10):
        start_idx = i * part_size
        if i == 9:  # para a última parte, incluir todas as linhas restantes
            df_part = df_aggregated[start_idx:]
        else:
            df_part = df_aggregated[start_idx:start_idx + part_size]

        # define o caminho para cada arquivo de parte
        part_file = os.path.join(full_output_folder, f'dados_parte_{i+1}.csv')
        
        # salva o DataFrame da parte em um novo arquivo CSV
        df_part.to_csv(part_file, index=False)
        print(f'Salvo: {part_file}')

    return df_aggregated

# pasta de entrada com csv brutos
input_folder = '/home/yuri/Desktop/Datasetp'

# pasta de sadía com csvs tratados
output_folder = '/home/yuri/Desktop/test'  
processarArquivosNaPasta(input_folder, output_folder)

# agrupa todos os csv em um só, dividindo em 10 partes iguais
output_subfolder = 'CSV Agegrado'
agregarArquivos(output_folder, output_subfolder)
