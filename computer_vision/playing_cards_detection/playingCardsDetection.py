from ultralytics import YOLO
from playingCardsFunc import findPokerHand
import cv2
import cvzone
import math
import torch


cap = cv2.VideoCapture("../videos/poker-1.mp4")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # chỉnh width cho webcam
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # chỉnh height cho webcam

classnames = [
    "10c","10d","10h","10s","2c","2d","2h","2s","3c","3d","3h","3s","4c","4d","4h","4s","5c","5d","5h","5s","6c","6d",
    "6h","6s","7c","7d","7h","7s","8c","8d","8h","8s","9c","9d","9h","9s","Ac","Ad","Ah","As",
    "Jc","Jd","Jh","Js","Kc","Kd","Kh","Ks","Qc","Qd","Qh","Qs"
]

model = YOLO('yolov8m_synthetic.pt') 

print(f"Python-{torch.__version__}")
if torch.cuda.is_available():
    print("GPU is available. Using:", torch.cuda.get_device_name(0))
else:
    print("GPU is NOT available. Using CPU.")
    
redColor = (0, 0, 255)
greenColor = (0, 255, 0)
blueColor = (255, 0, 0)
whiteColor = (255, 255, 255)

while True:
    success, img = cap.read() # đọc frame từ webcam
    results = model(img, stream=True) # dự đoán object trong frame, stream=True để đọc frame liên tục
    cardList = []
    for r in results:
        boxes = r.boxes # lấy tất cả box của các object
        for box in boxes:
            # Bonding Box
            x1, y1, x2, y2 = box.xyxy[0] # lấy tọa độ của box 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # chuyển tọa độ thành số nguyên 
            w, h = x2 - x1, y2 - y1 # tính chiều rộng và chiều cao của box
            
            # Check if the object is in the attentionArray
            cls = int(box.cls[0]) # lấy class name của box
            cardList.append(classnames[cls])
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100 # lấy độ tin cậy của box
       
            cv2.rectangle(img, (x1, y1), (x2, y2), greenColor, 2)
            cvzone.putTextRect(img, f"{classnames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=0.6, thickness=1, colorR=greenColor, colorT=whiteColor)
    
    result = findPokerHand(cardList)                   
    cvzone.putTextRect(img, f"Result: {result}", (50, 200), scale=4, thickness=1, colorR=greenColor, colorT=whiteColor)
            
            
    cv2.imshow("Image", img)
    cv2.waitKey(1) # chờ 1ms
