from ultralytics import YOLO

import cv2
import cvzone
import math
import torch


cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # chỉnh width cho webcam
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # chỉnh height cho webcam

classnames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 
              'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 
              'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck', 'truck and trailer', 
              'van', 'vehicle', 'wheel loader']
classnames_vi = ['Máy xúc', 'Găng tay', 'Mũ bảo hộ', 'Thang', 'Khẩu trang', 'Không mũ bảo hộ', 
                 'Không khẩu trang', 'Không áo phản quang', 'Người', 'SUV', 'Cọc an toàn', 'Áo phản quang',
                 'Xe buýt', 'Xe ben', 'Trụ cứu hỏa', 'Máy móc', 'Xe minivan', 'Xe sedan', 'Xe đầu kéo', 'Moóc',
                 'Xe tải', 'Xe tải kèm moóc', 'Xe van', 'Phương tiện', 'Xe xúc lật']

model = YOLO('ppe.pt') 

print(f"Python-{torch.__version__}")
if torch.cuda.is_available():
    print("GPU is available. Using:", torch.cuda.get_device_name(0))
else:
    print("GPU is NOT available. Using CPU.")
    
redColor = (0, 0, 255)
greenColor = (0, 255, 0)
blueColor = (255, 0, 0)
whiteColor = (255, 255, 255)

attentionArray = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Hardhat', 'Mask', 'Safety Vest', 'Person']
safetyArray = ['Hardhat', 'Mask', 'Safety Vest']
noSafetyArray = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

while True:
    success, img = cap.read() # đọc frame từ webcam
    results = model(img, stream=True) # dự đoán object trong frame, stream=True để đọc frame liên tục
    for r in results:
        boxes = r.boxes # lấy tất cả box của các object
        for box in boxes:
            # Bonding Box
            x1, y1, x2, y2 = box.xyxy[0] # lấy tọa độ của box 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # chuyển tọa độ thành số nguyên 
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # vẽ hình chữ nhật lên frame
            w, h = x2 - x1, y2 - y1 # tính chiều rộng và chiều cao của box
            
            # Check if the object is in the attentionArray
            cls = int(box.cls[0]) # lấy class name của box
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100 # lấy độ tin cậy của box
       
            if classnames[cls] in attentionArray:
                            if classnames[cls] in safetyArray:
                                cv2.rectangle(img, (x1, y1), (x2, y2), greenColor, 2)
                                cvzone.putTextRect(img, f"{classnames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=0.6, thickness=1, colorR=greenColor, colorT=whiteColor)
                            elif classnames[cls] in noSafetyArray:
                                cv2.rectangle(img, (x1, y1), (x2, y2), redColor, 2)
                                cvzone.putTextRect(img, f"{classnames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=0.6, thickness=1, colorR=redColor, colorT=whiteColor)   
                            else:
                                cv2.rectangle(img, (x1, y1), (x2, y2), blueColor, 2)
                                cvzone.putTextRect(img, f"{classnames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=0.6, thickness=1, colorR=blueColor, colorT=whiteColor)
            
            
    cv2.imshow("Image", img)
    cv2.waitKey(1) # chờ 1ms
