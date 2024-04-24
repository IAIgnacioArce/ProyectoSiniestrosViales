## FUNCIONES DE UTILIDAD PARA EL ETL Y EDA
# Importaciones
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def duplicados_por_columna(df, columna):
    '''
    Verifica y muestra filas duplicadas en un DataFrame basado en una columna específica.

    Esta función toma como entrada un DataFrame y el nombre de una columna específica.
    Luego, identifica las filas duplicadas basadas en el contenido de la columna especificada,
    las filtra y las ordena para una comparación más sencilla.

    Parameters:
        df (pandas.DataFrame): El DataFrame en el que se buscarán filas duplicadas.
        columna (str): El nombre de la columna basada en la cual se verificarán las duplicaciones.

    Returns:
        pandas.DataFrame or str: Un DataFrame que contiene las filas duplicadas filtradas y ordenadas,
        listas para su inspección y comparación, o el mensaje "No hay duplicados" si no se encuentran duplicados.
    '''
    # Se filtran las filas duplicadas
    duplicated_rows = df[df.duplicated(subset=columna, keep=False)]
    if duplicated_rows.empty:
        return "No hay duplicados"
    
    # se ordenan las filas duplicadas para comparar entre sí
    duplicated_rows_sorted = duplicated_rows.sort_values(by=columna)
    return duplicated_rows_sorted

def tipo_variable(df):
    '''
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
    '''

    mi_dict = {"nombre_campo": [], "tipo_datos": []}

    for columna in df.columns:
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
    df_info = pd.DataFrame(mi_dict)
        
    return df_info

def convertir_a_datatime(x):
    '''
    Convierte un valor a un objeto de tiempo (time) de Python si es posible.

    Esta función acepta diferentes tipos de entrada y trata de convertirlos en objetos de tiempo (time) de Python.
    Si la conversión no es posible, devuelve None.

    Parameters:
        x (str, datetime, or any): El valor que se desea convertir a un objeto de tiempo (time).

    Returns:
        datetime.time or None: Un objeto de tiempo (time) de Python si la conversión es exitosa,
        o None si no es posible realizar la conversión.
    '''
    if isinstance(x, str):
        try:
            return datetime.strptime(x, "%H:%M:%S").time()
        except ValueError:
            return None
    elif isinstance(x, datetime):
        return x.time()
    return x

def imputa_valor_frecuente(df, columna):
    '''
    Imputa los valores faltantes en una columna de un DataFrame con el valor más frecuente.

    Esta función reemplaza los valores "SD" con NaN en la columna especificada,
    luego calcula el valor más frecuente en esa columna y utiliza ese valor
    para imputar los valores faltantes (NaN).

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna a ser imputada.
        columna (str): El nombre de la columna en la que se realizará la imputación.

    Returns:
        None
    '''
    # Se reemplaza "SD" con NaN en la columna
    df[columna] = df[columna].replace('SD', pd.NA)

    # Se calcula el valor más frecuente en la columna
    valor_mas_frecuente = df[columna].mode().iloc[0]
    print(f'El valor mas frecuente es: {valor_mas_frecuente}')

    # Se imputan los valores NaN con el valor más frecuente
    df[columna].fillna(valor_mas_frecuente, inplace=True)
    
def imputa_edad_media_segun_sexo(df):
    '''
    Imputa valores faltantes en la columna 'Edad' utilizando la edad promedio según el género.

    Esta función reemplaza los valores "SD" con NaN en la columna 'Edad', calcula la edad promedio
    para cada grupo de género (Femenino y Masculino), imprime los promedios calculados y
    luego llena los valores faltantes en la columna 'Edad' utilizando el promedio correspondiente
    al género al que pertenece cada fila en el DataFrame.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna 'Edad' a ser imputada.

    Returns:
        None
    '''
    
    # Se reemplaza "SD" con NaN en la columna 'edad'
    df['Edad'] = df['Edad'].replace('SD', pd.NA)

    # Se calcula el promedio de edad para cada grupo de género
    promedio_por_genero = df.groupby('Sexo')['Edad'].mean()
    print(f'La edad promedio de Femenino es {round(promedio_por_genero["FEMENINO"])} y de Masculino es {round(promedio_por_genero["MASCULINO"])}')

    # Se llenan los valores NaN en la columna 'edad' utilizando el promedio correspondiente al género
    df['Edad'] = df.apply(lambda row: promedio_por_genero[row['Sexo']] if pd.isna(row['Edad']) else row['Edad'], axis=1)
    # Lo convierte a entero
    df['Edad'] = df['Edad'].astype(int)
    
def verificar_tipo_datos_y_nulos(df):
    '''
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna, el porcentaje de valores no nulos y nulos, así como la
    cantidad de valores nulos por columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
        - 'no_nulos_%': Porcentaje de valores no nulos en cada columna.
        - 'nulos_%': Porcentaje de valores nulos en cada columna.
        - 'nulos': Cantidad de valores nulos en cada columna.
    '''

    mi_dict = {"nombre_campo": [], "tipo_datos": [], "no_nulos_%": [], "nulos_%": [], "nulos": []}

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        mi_dict["nulos_%"].append(round(100-porcentaje_no_nulos, 2))
        mi_dict["nulos"].append(df[columna].isnull().sum())

    df_info = pd.DataFrame(mi_dict)
        
    return df_info
def victimas_por_anio(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['Año'], y=df['Cantidad víctimas'], estimator=sum, palette='pastel')
    plt.title('Cantidad de Víctimas por Año')
    plt.xlabel('Año')
    plt.ylabel('Cantidad de Víctimas')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def accidentes_mensuales(df):
    '''
    Crea gráficos de línea para la cantidad de víctimas de accidentes mensuales por año.

    Esta función toma un DataFrame que contiene datos de accidentes, extrae los años únicos
    presentes en la columna 'Año', y crea gráficos de línea para la cantidad de víctimas por mes
    para cada año. Los gráficos se organizan en una cuadrícula de subgráficos de 2x3.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene los datos de accidentes, con una columna 'Año'.

    Returns:
        None
    '''
    # Se obtiene una lista de años únicos
    años = df['Año'].unique()

    # Se define el número de filas y columnas para la cuadrícula de subgráficos
    n_filas = 3
    n_columnas = 2

    # Se crea una figura con subgráficos en una cuadrícula de 2x3
    fig, axes = plt.subplots(n_filas, n_columnas, figsize=(14, 8))

    # Se itera a través de los años y crea un gráfico por año
    for i, year in enumerate(años):
        fila = i // n_columnas
        columna = i % n_columnas
        
        # Se filtran los datos para el año actual y agrupa por mes
        data_mensual = (df[df['Año'] == year]
                        .groupby('Mes')
                        .agg({'Cantidad víctimas':'sum'}))
        
        # Se configura el subgráfico actual
        ax = axes[fila, columna]
        data_mensual.plot(ax=ax, kind='line')
        ax.set_title('Año ' + str(year)) ; ax.set_xlabel('Mes') ; ax.set_ylabel('Cantidad de Víctimas')
        ax.legend_ = None
        
    # Se muestra y acomoda el gráfico
    plt.tight_layout()
    plt.show()

def victimas_por_mes(df):    
    '''
    Crea un gráfico de barras que muestra la cantidad de víctimas de accidentes por mes.

    Esta función toma un DataFrame que contiene datos de accidentes, agrupa los datos por mes
    y calcula la cantidad total de víctimas por mes. Luego, crea un gráfico de barras que muestra
    la cantidad de víctimas para cada mes.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene los datos de accidentes con una columna 'Mes'.

    Returns:
        None
    '''
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['Mes'], y=df['Cantidad víctimas'], estimator=sum, palette='pastel', ci=None)
    plt.title('Cantidad de Víctimas por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad de Víctimas')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def cantidad_victimas_por_dia_semana(df):
    '''
    Crea un gráfico de barras que muestra la cantidad de víctimas de accidentes por día de la semana.

    Esta función toma un DataFrame que contiene datos de accidentes, convierte la columna 'Fecha' a tipo de dato
    datetime si aún no lo es, extrae el día de la semana (0 = lunes, 6 = domingo), mapea el número del día
    de la semana a su nombre, cuenta la cantidad de accidentes por día de la semana y crea un gráfico de barras
    que muestra la cantidad de víctimas para cada día de la semana.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene los datos de accidentes con una columna 'Fecha'.

    Returns:
        None
    '''
    # Se convierte la columna 'Fecha' a tipo de dato datetime si aún no lo es
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Se extrae el nombre del día de la semana (Lunes, Martes, ..., Domingo)
    df['Día Semana'] = df['Fecha'].dt.day_name(locale='es')
    
    # Se cuenta la cantidad de víctimas por día de la semana
    data = df.groupby('Día Semana')['Cantidad víctimas'].sum().reset_index()
    
    # Se crea el gráfico de barras con Seaborn
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Día Semana', y='Cantidad víctimas', data=data, palette='viridis')
    ax.set_title('Cantidad de Víctimas por Día de la Semana', fontsize=16)
    ax.set_xlabel('Día de la Semana', fontsize=14)
    ax.set_ylabel('Cantidad de Víctimas', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Se muestra el gráfico
    plt.tight_layout()
    plt.show()
    
    # Se muestra la información sobre la menor y mayor cantidad de víctimas en el output
    min_victimas = data['Cantidad víctimas'].min()
    max_victimas = data['Cantidad víctimas'].max()
    print(f'El día de la semana con menor cantidad de víctimas tiene {min_victimas} víctimas')
    print(f'El día de la semana con mayor cantidad de víctimas tiene {max_victimas} víctimas')
    print(f'La diferencia porcentual es de {round((max_victimas - min_victimas) / min_victimas * 100, 2)}%')

def crea_categoria_momento_dia(hora):
  """
  Devuelve la categoría de tiempo correspondiente a la hora proporcionada.

  Parameters:
    hora: La hora a clasificar.

  Returns:
    La categoría de tiempo correspondiente.
  """
  if hora.hour >= 6 and hora.hour <= 10:
    return "Mañana"
  elif hora.hour >= 11 and hora.hour <= 13:
    return "Medio día"
  elif hora.hour >= 14 and hora.hour <= 18:
    return "Tarde"
  elif hora.hour >= 19 and hora.hour <= 23:
    return "Noche"
  else:
    return "Madrugada"

def cantidad_accidentes_por_categoria_tiempo(df):
    '''
    Calcula la cantidad de accidentes por categoría de tiempo y muestra un gráfico de barras.

    Esta función toma un DataFrame que contiene una columna 'Hora' y utiliza la función
    'crea_categoria_momento_dia' para crear la columna 'Categoria tiempo'. Luego, cuenta
    la cantidad de accidentes por cada categoría de tiempo, calcula los porcentajes y
    genera un gráfico de barras que muestra la distribución de accidentes por categoría de tiempo.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la información de los accidentes.

    Returns:
        None
    '''
    # Se aplica la función crea_categoria_momento_dia para crear la columna 'categoria_tiempo'
    df['Categoria tiempo'] = df['Hora'].apply(crea_categoria_momento_dia)

    # Se cuenta la cantidad de accidentes por categoría de tiempo
    data = df['Categoria tiempo'].value_counts().reset_index()
    data.columns = ['Categoria tiempo', 'Cantidad accidentes']

    # Se calculan los porcentajes
    total_accidentes = data['Cantidad accidentes'].sum()
    data['Porcentaje'] = (data['Cantidad accidentes'] / total_accidentes) * 100
    
    # Se crea el gráfico de barras
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Categoria tiempo', y='Cantidad accidentes', data=data, palette='Set2')
    ax.set_title('Cantidad de Accidentes por Categoría de Tiempo', fontsize=16)
    ax.set_xlabel('Categoría de Tiempo', fontsize=14)
    ax.set_ylabel('Cantidad de Accidentes', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Se agrega las cantidades en las barras
    for index, row in data.iterrows():
        ax.text(index, row["Cantidad accidentes"], f'{row["Cantidad accidentes"]}', ha='center', va='bottom', fontsize=12)
    
    # Se muestra el gráfico
    plt.tight_layout()
    plt.show()

def cantidad_accidentes_por_horas_del_dia(df):
    '''
    Genera un gráfico de barras que muestra la cantidad de accidentes por hora del día.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    '''
    # Combinar la fecha y hora para crear objetos datetime completos
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha'].astype(str) + ' ' + df['Hora'].astype(str))

    # Extraer la hora del día de la columna 'Fecha_Hora'
    df['Hora del día'] = df['Fecha_Hora'].dt.hour

    # Contar la cantidad de accidentes por hora del día
    data = df['Hora del día'].value_counts().sort_index().reset_index()
    data.columns = ['Hora del día', 'Cantidad de accidentes']

    # Crear el gráfico de barras
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Hora del día', y='Cantidad de accidentes', data=data, palette='pastel', alpha=0.7)
    ax.set_title('Cantidad de Accidentes por Hora del Día', fontsize=16)
    ax.set_xlabel('Hora del día', fontsize=14)
    ax.set_ylabel('Cantidad de accidentes', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Agregar el valor de cada barra en el gráfico
    for index, row in data.iterrows():
        ax.text(row["Hora del día"], row["Cantidad de accidentes"], f'{row["Cantidad de accidentes"]}', ha='center', va='bottom', fontsize=10)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()

def cantidad_accidentes_semana_fin_de_semana(df):
    '''
    Genera un gráfico de barras que muestra la cantidad de accidentes por tipo de día (semana o fin de semana).

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    '''
    # Se convierte la columna 'fecha' a tipo de dato datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Se extrae el día de la semana (0 = lunes, 6 = domingo)
    df['Dia semana'] = df['Fecha'].dt.dayofweek
    
    # Se crea una columna 'tipo_dia' para diferenciar entre semana y fin de semana
    df['Tipo de día'] = df['Dia semana'].apply(lambda x: 'Fin de Semana' if x >= 5 else 'Semana')
    
    # Se cuenta la cantidad de accidentes por tipo de día
    data = df['Tipo de día'].value_counts().reset_index()
    data.columns = ['Tipo de día', 'Cantidad de accidentes']
    
    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x='Tipo de día', y='Cantidad de accidentes', data=data)
    
    ax.set_title('Cantidad de accidentes por tipo de día') ; ax.set_xlabel('Tipo de día') ; ax.set_ylabel('Cantidad de accidentes')
    
    # Se agrega las cantidades en las barras
    for index, row in data.iterrows():
        ax.annotate(f'{row["Cantidad de accidentes"]}', (index, row["Cantidad de accidentes"]), ha='center', va='bottom')
    
    # Se muestra el gráfico
    plt.show()

def distribucion_edad(df):
    '''
    Genera un gráfico con un histograma y un boxplot que muestran la distribución de la edad de los involucrados en los accidentes.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico con un histograma y un boxplot.
    '''
    # Crear una figura con un solo eje x compartido
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Graficar el histograma de la edad
    sns.histplot(df['Edad'], kde=True, ax=ax[0], color='skyblue', edgecolor='black', linewidth=1.5)
    ax[0].set_title('Distribución de Edad', fontsize=16) 
    ax[0].set_ylabel('Frecuencia', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    
    # Graficar el boxplot de la edad
    sns.boxplot(x=df['Edad'], ax=ax[1], color='salmon')
    ax[1].set_xlabel('Edad', fontsize=14)
    ax[1].set_ylabel('')  # Eliminar el nombre del eje y del boxplot
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    
    # Ajustar y mostrar el gráfico
    plt.tight_layout()
    plt.show()
    
def distribucion_edad_por_anio(df):
    '''
    Genera un gráfico de boxplot que muestra la distribución de la edad de las víctimas de accidentes por año.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de boxplot.
    '''
    # Se crea el gráfico de boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Año', y='Edad', data=df)
    
    plt.title('Boxplot de Edades de Víctimas por Año') ; plt.xlabel('Año') ; plt.ylabel('Edad de las Víctimas')
     
    # Se muestra el gráfico
    plt.show()

def cantidades_accidentes_por_anio_y_sexo(df):
    '''
    Genera un gráfico de barras que muestra la cantidad de accidentes por año y sexo.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    '''
    # Se crea el gráfico de barras
    plt.figure(figsize=(12, 4))
    sns.barplot(x='Año', y='Edad', hue='Sexo', data=df,)
    
    plt.title('Cantidad de Accidentes por Año y Sexo')
    plt.xlabel('Año') ; plt.ylabel('Edad de las víctimas') ; plt.legend(title='Sexo')
    
    # Se muestra el gráfico
    plt.show()
    
def cohen(group1, group2):
    '''
    Calcula el tamaño del efecto de Cohen d para dos grupos.

    Parameters:
        grupo1: El primer grupo.
        grupo2: El segundo grupo.

    Returns:
        El tamaño del efecto de Cohen d.
    '''
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1, n2 = len(group1), len(group2)
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d

def cohen_por_año(df):
    '''
    Calcula el tamaño del efecto de Cohen d para dos grupos para los años del Dataframe.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        El tamaño del efecto de Cohen d.
    '''
    # Se obtienen los años del conjunto de datos
    años_unicos = df['Año'].unique()
    # Se crea una lista vacía para guardar los valores de Cohen
    cohen_lista = []
    # Se itera por los años y se guarda Cohen para cada grupo
    for a in años_unicos:
        grupo1 = df[((df['Sexo'] == 'MASCULINO') & (df['Año'] == a))]['Edad']
        grupo2 = df[((df['Sexo'] == 'FEMENINO')& (df['Año'] == a))]['Edad']
        d = cohen(grupo1, grupo2)
        cohen_lista.append(d)

    # Se crea un Dataframe
    cohen_df = pd.DataFrame()
    cohen_df['Año'] = años_unicos
    cohen_df['Estadistico de Cohen'] = cohen_lista
    cohen_df
    
    # Se grafica los valores de Cohen para los años
    plt.figure(figsize=(8, 4))
    plt.bar(cohen_df['Año'], cohen_df['Estadistico de Cohen'], color='skyblue')
    plt.xlabel('Año') ; plt.ylabel('Estadístico de Cohen') ; plt.title('Estadístico de Cohen por Año')
    plt.xticks(años_unicos)
    plt.show()

def edad_y_rol_victimas(df):
    '''
    Genera un gráfico de la distribución de la edad de las víctimas por rol.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Edad', y='Rol', data=df, palette='pastel', linewidth=1.5)
    plt.title('Distribución de Edades por Rol de las Víctimas')
    plt.xlabel('Edad')
    plt.ylabel('Rol')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    
def distribucion_edad_por_victima(df):
    '''
    Genera un gráfico de la distribución de la edad de las víctimas por tipo de vehículo.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(x='Víctima', y='Edad', data=df, palette='Set2', linewidth=1.5)
    
    plt.title('Distribución de Edades de Víctimas por Tipo de Vehículo') 
    plt.xlabel('Tipo de Vehículo') 
    plt.ylabel('Edad de las Víctimas')
    plt.xticks(rotation=45)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()
    
def cantidad_accidentes_sexo(df):
    '''
    Genera un resumen de la cantidad de accidentes por sexo de los conductores.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad de accidentes por sexo de los conductores en orden descendente.
    * Un DataFrame que muestra la cantidad y el porcentaje de accidentes por sexo de los conductores.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    # Se convierte la columna 'fecha' a tipo de dato datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Se extrae el día de la semana (0 = lunes, 6 = domingo)
    df['Dia semana'] = df['Fecha'].dt.dayofweek
    
    # Se crea una columna 'tipo_dia' para diferenciar entre semana y fin de semana
    df['Tipo de día'] = df['Dia semana'].apply(lambda x: 'Fin de Semana' if x >= 5 else 'Semana')
    
    # Se cuenta la cantidad de accidentes por tipo de día
    data = df['Tipo de día'].value_counts().reset_index()
    data.columns = ['Tipo de día', 'Cantidad de accidentes']
    
    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x='Tipo de día', y='Cantidad de accidentes', data=data)
    
    ax.set_title('Cantidad de accidentes por tipo de día') ; ax.set_xlabel('Tipo de día') ; ax.set_ylabel('Cantidad de accidentes')
    
    # Se agrega las cantidades en las barras
    for index, row in data.iterrows():
        ax.annotate(f'{row["Cantidad de accidentes"]}', (index, row["Cantidad de accidentes"]), ha='center', va='bottom')
    
    # Se muestra el gráfico
    plt.show()

def cantidad_victimas_sexo_rol_victima(df):
    '''
    Genera un resumen de la cantidad de víctimas por sexo, rol y tipo de vehículo en un accidente de tráfico.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Gráficos de barras que muestran la cantidad de víctimas por sexo, rol y tipo de vehículo en orden descendente.
    * DataFrames que muestran la cantidad y el porcentaje de víctimas por sexo, rol y tipo de vehículo.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    # Se crea el gráfico
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Gráfico 1: Sexo
    sns.countplot(data=df, x='Sexo', ax=axes[0])
    axes[0].set_title('Cantidad de víctimas por sexo') ; axes[0].set_ylabel('Cantidad de víctimas')

    # Se define una paleta de colores personalizada (invierte los colores)
    colores_por_defecto = sns.color_palette()
    colores_invertidos = [colores_por_defecto[1], colores_por_defecto[0]]
    
    # Gráfico 2: Rol
    df_rol = df.groupby(['Rol', 'Sexo']).size().unstack(fill_value=0)
    df_rol.plot(kind='bar', stacked=True, ax=axes[1], color=colores_invertidos)
    axes[1].set_title('Cantidad de víctimas por rol') ; axes[1].set_ylabel('Cantidad de víctimas') ; axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend().set_visible(False)
    
    # Gráfico 3: Tipo de vehículo
    df_victima = df.groupby(['Víctima', 'Sexo']).size().unstack(fill_value=0)
    df_victima.plot(kind='bar', stacked=True, ax=axes[2], color=colores_invertidos)
    axes[2].set_title('Cantidad de víctimas por tipo de vehículo') ; axes[2].set_ylabel('Cantidad de víctimas') ; axes[2].tick_params(axis='x', rotation=45)
    axes[2].legend().set_visible(False)

    # Se muestran los gráficos
    plt.show()
    

def cantidad_victimas_participantes(df):
    '''
    Genera un resumen de la cantidad de víctimas por número de participantes en un accidente de tráfico.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad de víctimas por número de participantes en orden descendente.
    * Un DataFrame que muestra la cantidad y el porcentaje de víctimas por número de participantes.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    # Se ordenan los datos por 'Participantes' en orden descendente por cantidad
    ordenado = df['Participantes'].value_counts().reset_index()
    ordenado = ordenado.rename(columns={'Cantidad': 'participantes'})
    ordenado = ordenado.sort_values(by='count', ascending=False)
    
    plt.figure(figsize=(15, 4))
    
    # Se crea el gráfico de barras
    ax = sns.barplot(data=ordenado, x='Participantes', y='count', order=ordenado['Participantes'])
    ax.set_title('Cantidad de víctimas por participantes')
    ax.set_ylabel('Cantidad de víctimas')
    # Rotar las etiquetas del eje x a 45 grados
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Se muestra el gráfico
    plt.show()
    
    # # Se calcula la cantidad de víctimas por participantes
    # participantes_counts = df['Participantes'].value_counts().reset_index()
    # participantes_counts.columns = ['Participantes', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por participantes
    # total_victimas = participantes_counts['Cantidad de víctimas'].sum()
    # participantes_counts['Porcentaje de víctimas'] = round((participantes_counts['Cantidad de víctimas'] / total_victimas) * 100,2)

    # # Se ordenan los datos por cantidad de víctimas en orden descendente
    # participantes_counts = participantes_counts.sort_values(by='Cantidad de víctimas', ascending=False)
    
    # # Se imprimen resumenes
    # print("Resumen de víctimas por participantes:")
    # print(participantes_counts)
    
def cantidad_acusados(df):
    '''
    Genera un resumen de la cantidad de acusados en un accidente de tráfico.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad de acusados en orden descendente.
    * Un DataFrame que muestra la cantidad y el porcentaje de acusados.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    # Se ordenan los datos por 'Participantes' en orden descendente por cantidad
    ordenado = df['Acusado'].value_counts().reset_index()
    ordenado = ordenado.rename(columns={'Cantidad': 'Acusado'})
    ordenado = ordenado.sort_values(by='count', ascending=False)
    
    plt.figure(figsize=(15, 4))
    
    # Crear el gráfico de barras
    ax = sns.barplot(data=ordenado, x='Acusado', y='count', order=ordenado['Acusado'])
    ax.set_title('Cantidad de acusados en los hechos') ; ax.set_ylabel('Cantidad de acusados') 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Se muestra el gráfico
    plt.show()
    
    # # Se calcula la cantidad de acusados
    # acusados_counts = df['Acusado'].value_counts().reset_index()
    # acusados_counts.columns = ['Acusado', 'Cantidad de acusados']

    # # Se calcula el porcentaje de acusados
    # total_acusados = acusados_counts['Cantidad de acusados'].sum()
    # acusados_counts['Porcentaje de acusados'] = round((acusados_counts['Cantidad de acusados'] / total_acusados) * 100,2)

    # # Se ordenan los datos por cantidad de acusados en orden descendente
    # acusados_counts = acusados_counts.sort_values(by='Cantidad de acusados', ascending=False)
    # # Se imprimen resumen
    # print("Resumen de acusados:")
    # print(acusados_counts)

def accidentes_tipo_de_calle(df):
    '''
    Genera un resumen de los accidentes de tráfico por tipo de calle y cruce.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad de víctimas por tipo de calle.
    * Un gráfico de barras que muestra la cantidad de víctimas en cruces.
    * Un DataFrame que muestra la cantidad y el porcentaje de víctimas por tipo de calle.
    * Un DataFrame que muestra la cantidad y el porcentaje de víctimas en cruces.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    '''
    plt.figure(figsize=(12, 6))

    # Gráfico de barras para el tipo de calle
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='Tipo de calle', palette='viridis')
    plt.title('Cantidad de víctimas por tipo de calle')
    plt.xlabel('Tipo de calle')
    plt.ylabel('Cantidad de víctimas')
    plt.xticks(rotation=45, ha='right')

    # Gráfico de barras para el cruce
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='Cruce', palette='viridis')
    plt.title('Cantidad de víctimas en cruces')
    plt.xlabel('Cruce')
    plt.ylabel('Cantidad de víctimas')
    plt.xticks(rotation=45, ha='right')

    # Ajustar diseño
    plt.tight_layout()
    plt.show()
    # # Se calcula la cantidad de víctimas por tipo de calle
    # tipo_calle_counts = df['Tipo de calle'].value_counts().reset_index()
    # tipo_calle_counts.columns = ['Tipo de calle', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por tipo de calle
    # tipo_calle_counts['Porcentaje de víctimas'] = round((tipo_calle_counts['Cantidad de víctimas'] / tipo_calle_counts['Cantidad de víctimas'].sum()) * 100,2)

    # # Se calcula la cantidad de víctimas por cruce
    # cruce_counts = df['Cruce'].value_counts().reset_index()
    # cruce_counts.columns = ['Cruce', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por cruce
    # cruce_counts['Porcentaje de víctimas'] = round((cruce_counts['Cantidad de víctimas'] / cruce_counts['Cantidad de víctimas'].sum()) * 100,2)

    # # Se crean DataFrames para tipo de calle y cruce
    # df_tipo_calle = pd.DataFrame(tipo_calle_counts)
    # df_cruce = pd.DataFrame(cruce_counts)

    # #  Se muestran los DataFrames resultantes
    # print("Resumen por Tipo de Calle:")
    # print(df_tipo_calle)
    # print("\nResumen por Cruce:")
    # print(df_cruce)