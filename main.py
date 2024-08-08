import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import socket
import requests
from ip2geotools.databases.noncommercial import DbIpCity
from geopy.distance import distance

def printDetails(ip):
    res = DbIpCity.get(ip, api_key="free")
    #print(f"IP Address: {res.ip_address}")
    #print(f"Location: {res.city}, {res.region}, {res.country}")
    print(f"Coordinates: (Lat: {res.latitude}, Lng: {res.longitude})")




model=YOLO(input("""Enter the model you want to test:
best.pt
best2.pt
best3.pt
"""))


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
            
        



cap=cv2.VideoCapture(input("""Enter the video you would like to see: 
crash1.mp4
crash2.mp4
crash3.mp4
crash4.mp4
crash5.mp4
crash6.mp4
crash7.mp4
cr.mp4
> """))


my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
print(class_list)



count=0

while True:    
    ret,frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
   

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    #print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    #print(boxes)

    #print(boxes.data)
    
   
    #print(px)
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]

        if(c=="accident" ):
            print("Accident has happend please see the coordinates and ambulance should go there right away")

            
            ip_add = "172.25.248.127" #input("Enter IP: ")  # 198.35.26.96
            printDetails(ip_add)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
        cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
            
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()  
cv2.destroyAllWindows()




