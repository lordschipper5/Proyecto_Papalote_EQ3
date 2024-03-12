# PaploteAI
Sistema dedicado a solucionar el problema de Papalote Museo del Niño relacionado a la falta de recopilacion de datos como la edad y el genero del visistante. A continuacion se mencionan los archivos importantes con una descripcion y su necesidad en el proyecto.

# Overview
Este proyecto es un sistema de detección y reconocimiento de rostros que identifica el género y la edad de las personas en una transmisión de video. Utiliza OpenCV y DNN (Deep Neural Networks) para la detección de rostros y la predicción de edad/género. Los datos de edad y sexo luego se almacenan en una base de datos.

El proyecto utiliza varios modelos previamente entrenados:

# Archivos del modelo
opencv_face_detector.pbtxt y opencv_face_detector_uint8.pb para detección de rostros.
age_deploy.prototxt y age_net.caffemodel para predicción de edad.
Gender_deploy.prototxt y Gender_net.caffemodel para predicción de género.

Los principales guiones del proyecto son:

# Archivos de la implementacion
**Papalote_proyecto.py**: Este script captura video, detecta rostros y predice edad y género. También realiza reconocimiento facial para evitar la redundancia de datos. (Es la version inicial del proyecto sin la toma de filtros).
**server_model.py**: este es el script principal donde se captura videos, detecta rostros y predice la edad y el sexo. Además, realiza un seguimiento de la edad y el sexo más detectados y publica estos datos en una base de datos. También aplica una imagen de fondo específica basada en el género más detectado. (*Este es el servidor principal correr este archivo para ejecutar el proyecto*)
El proyecto también se conecta a una base de datos SQL Server usando SQLAlchemy y pyodbc, como se ve en server_model.py. Los detalles de la conexión de la base de datos se obtienen de las variables de entorno.

Para instalarlo se requieren las siguientes dependencias --> 
    - OpenCV
    - SQLAlchemy
    - PYODBC
    - Ultralytics
    - FastAPI
    - Python-dotenv
