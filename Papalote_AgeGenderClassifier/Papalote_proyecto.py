import cv2
import face_recognition
import numpy as np
import argparse
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL
from fastapi import HTTPException 
from fastapi.responses import JSONResponse
import logging

API_SERVER_PORT = 3010
DB_SERVER = "serverpapalote.database.windows.net"
DB_SERVER_DATABASE_NAME = "ProyectoPapaloteDatabase"
DB_SERVER_USER = "papalote2024"
DB_SERVER_PASS = "ProyectoEQ3"

# Utilizar las variables env
load_dotenv()
port = int(os.getenv('API_SERVER_PORT'))
driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv('DB_SERVER')
database = os.getenv('DB_SERVER_DATABASE_NAME')
username = os.getenv('DB_SERVER_USER')
password = os.getenv('DB_SERVER_PASS')

# Conectar a base de datos
odbc_conn = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'
connection_url = URL.create(
    drivername="mssql+pyodbc",
    query={
        "odbc_connect": odbc_conn
    }
)

engine = create_engine(connection_url)
cursor = engine.connect()

#Para pruebas
def res(status: int, success: bool, data: any):
    content = {'success': success, 'data': data}
    return JSONResponse(content=content, status_code=status)

#Funcion para insertar en la base de datos
def post_db(gender: str, age: str):
    conn = None
    cursor = None

    try:
        conn = engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Visitantes (Gender, Age) VALUES (?, ?)", (gender, age))
        conn.commit()
        return res(status=200, success=True, data={'message': 'Visitor added to the db successfully'})

    except Exception as e:
        logging.error(f'{e}')
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    finally:
        #Cerrar conexiones
        if cursor:
            cursor.close()
        if conn:
            conn.close()

#Detectar edad y genero
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

#Reconocimiento facial a visitantes detectados
def face_recon(frame,faceBox,known_face_encodings):
    face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),
        max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

    face_locations = [(max(0,faceBox[1]-padding),min(faceBox[2]+padding, frame.shape[1]-1),min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding))]

    face_encoding = face_recognition.face_encodings(frame,face_locations)[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)

    if not True in matches:
        known_face_encodings.append(face_encoding)            
        print('nuevo registro')
    if True in matches:
        first_match_index = matches.index(True)
        print('registro existente')
    return face,known_face_encodings,first_match_index

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)','(4-6)','(8-12)','(15-20)', '(25-32)','(38-43)', '(48-53)', '(60-100)']
genderList=['Hombre','Mujer']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20

known_face_encodings = []

while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()

    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if faceBoxes:
        for faceBox in faceBoxes:

            face, known_face_encodings, match_index = face_recon(frame,faceBox,known_face_encodings)
                
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]

            

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Detectando edad y genero", resultImg)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
post_db(str(gender), str(age[1:-1]))

