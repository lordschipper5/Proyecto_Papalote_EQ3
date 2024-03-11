import cv2
import argparse
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL
from fastapi import HTTPException 
from fastapi.responses import JSONResponse
import logging
import numpy as np
from ultralytics import YOLO
import uuid

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

# Detectar edad y genero
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
ageList=['(0-5)', '(6-12)', '(13-15)', '(16-19)', '(20-24)', '(25-29)', '(30-34)', '(35-39)', '(40-44)', '(45-49)', '(50-54)', '(55-59)', '(60-100)']
genderList=['Hombre','Mujer']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20

# Variables para guardar la edad y genero mas detectada
most_detected_age = None
most_detected_gender = None
max_age_count = 0
max_gender_count = 0

while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No se ha detectado ninguna cara")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Genero: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Edad: {age[1:-1]} Años')

        # Updatear el genero y edad mas detectados
        age_count = ageList.count(age)
        if age_count > max_age_count:
            max_age_count = age_count
            most_detected_age = age

        gender_count = genderList.count(gender)
        if gender_count > max_gender_count:
            max_gender_count = gender_count
            most_detected_gender = gender

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detectando edad y genero", resultImg)
    #Mandar a la DB

post_db(str(most_detected_gender), str(most_detected_age[1:-1]))

if most_detected_gender == 'Hombre':
    background = cv2.imread(r'masosauros.jpg')
else: 
    background = cv2.imread(r'butterfly.jpg')

video = cv2.VideoCapture(0)
hasFrame,frame=video.read()

model=YOLO('yolov8n-seg.pt')

results=model(frame)
h, w, _ = frame.shape
background = cv2.resize(background,(w,h))
big_mask = segmented_frame=np.zeros((h,w))
for result in results:
    i=0
    for mask in result.masks.data:
        if result.boxes.cls[i]==0:
            mask=mask.numpy()
            print(mask.shape)
            mask =cv2.resize(mask,(w,h))
            print(mask.shape)
            big_mask+=mask
            i+=1

red_channel = big_mask*frame[:,:,0]
blue_channel = big_mask*frame[:,:,1]
green_channel = big_mask*frame[:,:,2]

segmented_frame = np.zeros((h,w,3))
segmented_frame[:,:,0] = red_channel
segmented_frame[:,:,1] = blue_channel
segmented_frame[:,:,2] = green_channel

inverted_mask=(big_mask-1)*-1
red_channel_bg = inverted_mask*background[:,:,0]
blue_channel_bg = inverted_mask*background[:,:,1]
green_channel_bg = inverted_mask*background[:,:,2]

segmented_background=np.zeros((h,w,3))
segmented_background[:,:,0]=red_channel_bg
segmented_background[:,:,1]=blue_channel_bg
segmented_background[:,:,2]=green_channel_bg

final_image = segmented_frame + segmented_background
filename = str(uuid.uuid4()) + '.jpg'
cv2.imwrite(filename, final_image)
video.release()