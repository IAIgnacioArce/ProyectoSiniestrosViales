# Proyecto: Análisis de Homicidios en Siniestros Viales en CABA

## Introducción
Este proyecto de análisis de datos simula el rol de un Data Analyst Senior en el equipo de analistas de datos de una empresa consultora contratada por el Observatorio de Movilidad y Seguridad Vial (OMSV), perteneciente a la Secretaría de Transporte del Gobierno de la Ciudad Autónoma de Buenos Aires (CABA). El objetivo es proporcionar información para que las autoridades locales puedan tomar medidas efectivas para reducir la cantidad de víctimas fatales en siniestros viales en CABA. Se emplea un conjunto de datos sobre homicidios en siniestros viales ocurridos en la Ciudad de Buenos Aires durante el periodo 2016-2021.

## Contexto
Los siniestros viales, también conocidos como accidentes de tráfico, son eventos que involucran vehículos en las vías públicas y pueden tener diversas causas y consecuencias, desde daños materiales hasta lesiones graves o fatales. En CABA, estos siniestros son una preocupación importante debido al alto volumen de tráfico y la densidad poblacional.

Actualmente, la población de CABA es de aproximadamente 3,120,612 habitantes en una superficie de 200 km², lo que implica una densidad de aproximadamente 15,603 habitantes por km². Además, en julio de 2023, se registraron 12,437,735 vehículos transitando por los peajes de las autopistas de acceso a CABA. Por lo tanto, la prevención de siniestros viales y la implementación de políticas efectivas son cruciales para abordar este problema.

## Datos
Se utilizó una Base de Datos de Víctimas Fatales en Siniestros Viales, en formato Excel, que contiene dos pestañas de datos:

### HECHOS
Contiene información temporal, espacial y de participantes asociadas a cada hecho.

### VICTIMAS
Contiene información de cada víctima, incluyendo edad, sexo y modo de desplazamiento.

## Tecnologías Utilizadas
Para este proyecto se emplearon Python y Pandas para el procesamiento y análisis de datos, así como webscraping con BeautifulSoup para obtener datos complementarios, como la población en el año 2021. Además, se utilizó Power BI para la creación de un dashboard interactivo.

## ETL y EDA
Se realizó un proceso de Extracción, Transformación y Carga (ETL) de los datos, donde se estandarizaron nombres de variables, se analizaron nulos y duplicados, y se eliminaron columnas redundantes. Luego, se llevó a cabo un Análisis Exploratorio de Datos (EDA) para identificar patrones y tendencias.

## Análisis de los Datos
Se analizó la distribución temporal, perfil de las víctimas, roles y medios de transporte involucrados, así como la distribución espacial de los hechos. Se identificaron patrones relevantes que pueden informar la toma de decisiones.

## KPIs (Indicadores Clave de Desempeño)
En base al análisis realizado, se plantearon tres objetivos y KPIs asociados:

### Reducción de la Tasa de Homicidios en Siniestros Viales
Se propone reducir en un 10% la tasa de homicidios en siniestros viales de los últimos seis meses en comparación con el semestre anterior.

### Reducción de Accidentes Mortales de Motociclistas
Se plantea reducir en un 7% la cantidad de accidentes mortales de motociclistas en el último año respecto al año anterior.

### Reducción de la Tasa de Homicidios en Avenidas
Se busca reducir en un 10% la tasa de homicidios en las avenidas en el último año en comparación con el año anterior.

## Conclusiones y recomendaciones

Durante el periodo de análisis, se registraron un total de 696 víctimas fatales en accidentes de tránsito. La mayoría de estos incidentes, específicamente el 71%, ocurrieron durante los días de semana. En cuanto a la franja horaria, el 29% de los incidentes mortales tuvieron lugar por la mañana. Se destaca que el mes de Diciembre presenta el mayor número de víctimas en el período examinado.

El análisis demográfico revela que el 77% de las víctimas fueron hombres, y el 80% se encontraba en la edad adulta. Respecto al tipo de usuario, el 45% de las víctimas eran conductores de motocicletas. Además, el 76% de los incidentes mortales ocurrieron en cruces de calles, siendo las avenidas el escenario del 62% de los homicidios, con el 82% de ellos en intersecciones con otras calles.

En cuanto a los objetivos planteados, se logró cumplir con la meta de reducir la tasa de homicidios en el segundo semestre de 2021. Sin embargo, no se alcanzaron los objetivos de disminuir la cantidad de accidentes mortales en motociclistas ni en avenidas para el año 2021 en comparación con 2020.

Basándonos en los hallazgos anteriores, se sugieren las siguientes recomendaciones:

* Continuar monitoreando los objetivos establecidos y acompañarlos con campañas específicas, especialmente dirigidas a conductores de motocicletas y usuarios de avenidas.
* Reforzar las campañas de seguridad vial durante los días comprendidos entre viernes y lunes, haciendo hincapié particular en el mes de Diciembre.
* Implementar campañas específicas de conducción segura en avenidas y en los cruces de calles.
* Dirigir las campañas de seguridad hacia el sexo masculino, especialmente enfocadas en la conducción de motocicletas.