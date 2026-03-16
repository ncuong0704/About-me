from ultralytics import YOLO

import cv2
import cvzone
import math
import torch
from sort import *


# cap = cv2.VideoCapture(0) # 0 là webcam của máy, 1 là camera ngoài
cap = cv2.VideoCapture('../videos/payment-counter.mp4') # đọc video từ file
mask = cv2.imread('mask.png')
areaStartLeft = np.array([
   [687, 110], [801, 110], [845, 271], [702, 271]
], np.int32)
areaStartLeft = areaStartLeft.reshape((-1, 1, 2))
lineEndLeft = [710, 440, 958, 440]

areaStartRight = np.array([
   [967, 110], [1130, 110], [1230, 271], [1060, 271]
], np.int32)
areaStartRight = areaStartRight.reshape((-1, 1, 2))
lineEndRight = [1132, 343, 1280, 343]

listIdLeft = []
listIdRight = []
listIdDoneLeft = []
listIdDoneRight = []

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # chỉnh width cho webcam
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # chỉnh height cho webcam

classnames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
              "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
              "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
classnames_vi = [
    "nguoi", "xe dap", "xe hoi", "xe may", "may bay", "xe buyt", "tau hoa", "xe tai", "thuyen",
    "den giao thong", "tru nuoc chay", "bien dung", "dong ho do xe", "ghe dai", "chim", "meo",
    "cho", "ngua", "cuu", "bo", "voi", "gau", "ngua van", "huou cao co", "ba lo",
    "o", "tui xach", "ca vat", "vali", "dia nem", "van truot tuyet", "van truot tuyet bang", "bong the thao",
    "dieu", "gay bong chay", "gang tay bong chay", "van truot", "van luot song", "vot tennis",
    "chai", "ly ruou", "coc", "nia", "dao", "muong", "to", "chuoi", "tao",
    "banh mi kep", "cam", "bong cai xanh", "ca rot", "xuc xich", "pizza", "banh donut", "banh kem", "ghe",
    "ghe truong ky", "chau cay canh", "giuong", "ban an", "bon cau", "tivi", "may tinh xach tay", "chuot may tinh", "dieu khien",
    "ban phim", "dien thoai", "lo vi song", "lo nuong", "may nuong banh my", "bon rua", "tu lanh", "sach", "dong ho",
    "binh hoa", "keo", "gau bong", "may say toc", "ban chai danh rang"
]
# classnames lấy từ dataset COCO gồm 80 classes

model = YOLO('../YOLO-Weights/yolo11n.pt') # tải model từ ultralytics

print(f"Python-{torch.__version__}")
if torch.cuda.is_available():
    print("GPU is available. Using:", torch.cuda.get_device_name(0))
else:
    print("GPU is NOT available. Using CPU.")

# Tracking
# max_age giống như số lần mà bạn cho phép một vật bị "mất dấu" (tức là camera không nhìn thấy) trước khi xóa vật đó khỏi danh sách theo dõi.
# Nếu max_age = 20 thì nếu vật nào đó bị khuất trong tối đa 20 khung hình, chương trình vẫn nhớ nó, sau đó mới xóa.
# min_hits giống như số lần mà chương trình cần "gặp lại" một vật để chắc chắn đó là vật mới xuất hiện thật, không phải do nhầm lẫn.
# Nếu min_hits = 3 thì một vật phải xuất hiện ít nhất 3 lần liên tiếp thì mới được coi là "được theo dõi".
# iou_threshold: ngưỡng IOU
# Nếu iou_threshold = 0.3 thì nếu hai box có IOU lớn hơn 0.3 thì chương trình sẽ coi chúng là cùng một vật.
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    success, img = cap.read() # đọc frame từ webcam
    print(img.shape)
    if not success or img is None:
        break
    if img.shape[:2] != mask.shape[:2]:
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    else:
        mask_resized = mask
    imgRegion = cv2.bitwise_and(img, mask_resized) # áp dụng mask lên frame
    results = model(imgRegion, stream=True) # dự đoán object trong frame, stream=True để đọc frame liên tục
    detections = np.empty((0, 5)) # lưu trữ kết quả dự đoán
    for r in results:
        boxes = r.boxes # lấy tất cả box của các object
        for box in boxes:
            # Bonding Box
            x1, y1, x2, y2 = box.xyxy[0] # lấy tọa độ của box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # chuyển tọa độ thành số nguyên
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # vẽ hình chữ nhật lên frame
            w, h = x2 - x1, y2 - y1 # tính chiều rộng và chiều cao của box
            # cvzone.cornerRect(img, (x1, y1, w, h), l=5, t=2, rt=5) # vẽ hình chữ nhật lên frame
            #l=5 là độ dài của các đường viền của box
            #t=2 là độ dày của các đường viền của box
            #c=0 là màu của các đường viền của box
            #rt=5 là độ bo tròn của các đường viền của box

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100 # lấy độ tin cậy của box

            # Class Name
            cls = int(box.cls[0]) # lấy class name của box
            arr_class = ["nguoi"]
            if classnames_vi[cls] in arr_class and conf > 0.3:
                # Vẽ text lên box
                # cvzone.putTextRect(img, f"{classnames_vi[cls]} {conf}", (max(0, x1), max(35, y1)),
                #                 scale=0.6, thickness=1, offset=5) # max(0, x1), max(35, y1) để tránh vượt quá biên của frame
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))


    result_trackers = tracker.update(detections) # result_trackers là một mảng 2 chiều, mỗi hàng là một box, mỗi cột là: x1, y1, x2, y2, conf, id
    cv2.polylines(img, [areaStartLeft], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(img, [areaStartRight], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.line(img, (lineEndLeft[0], lineEndLeft[1]), (lineEndLeft[2], lineEndLeft[3]), (0, 0, 255), 5)
    cv2.line(img, (lineEndRight[0], lineEndRight[1]), (lineEndRight[2], lineEndRight[3]), (0, 0, 255), 5)
    for result in result_trackers:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w , h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + (h // 2) # tọa độ của tâm của box

        if cv2.pointPolygonTest(areaStartLeft, (cx, cy), False) >= 0:
            if id not in listIdLeft:
                listIdLeft.append(id)
        if id in listIdLeft:
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED) # vẽ điểm lên frame
            cvzone.cornerRect(img, (x1, y1, w, h), l=5, t=2, rt=5, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1)),
                                scale=1, thickness=2, offset=5, colorT=(255, 255, 255))

        if lineEndLeft[0] < cx < lineEndLeft[2] and lineEndLeft[1] - 15 < cy < lineEndLeft[1] + 15:
            if id in listIdLeft:
                listIdLeft.remove(id)
                listIdDoneLeft.append(id)
        
        if cv2.pointPolygonTest(areaStartRight, (cx, cy), False) >= 0:
            if id not in listIdRight:
                listIdRight.append(id)
        if id in listIdRight:
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED) # vẽ điểm lên frame
            cvzone.cornerRect(img, (x1, y1, w, h), l=5, t=2, rt=5, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1)),
                                scale=1, thickness=2, offset=5, colorT=(255, 255, 255))
        if lineEndRight[0] < cx < lineEndRight[2] and lineEndRight[1] - 15 < cy < lineEndRight[1] + 15:
            if id in listIdRight:
                listIdRight.remove(id)
                listIdDoneRight.append(id)

    # cvzone.putTextRect(img, f"Count: {len(total_id)}", (50, 50), scale=2, thickness=2, offset=5, colorT=(0, 0, 255))
    cv2.putText(img, f"{len(listIdDoneLeft)}", (546,491), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    cv2.putText(img, f"{len(listIdDoneRight)}", (1048,491), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    cv2.putText(img, f"Total: {len(listIdDoneLeft) + len(listIdDoneRight)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(1)
