from ultralytics import YOLO

import cv2
import cvzone
import math
import torch
from sort import *


# cap = cv2.VideoCapture(0) # 0 là webcam của máy, 1 là camera ngoài
cap = cv2.VideoCapture('../videos/cars.mp4') # đọc video từ file
mask = cv2.imread('mask.png')
limits = [210, 200, 420, 200] # tọa độ của đường giới hạn
total_id = []
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

# Đọc sẵn ảnh overlay xe và đảm bảo có kênh alpha, kích thước hợp lệ
img_car = cv2.imread('car.png')
if img_car is None:
    raise FileNotFoundError("Không tìm thấy file 'img_car.png' trong thư mục hiện tại.")

img_car = cv2.resize(img_car, (120, 60))
# Nếu ảnh chỉ có 3 kênh (BGR) thì thêm kênh alpha đầy đủ
if len(img_car.shape) == 3 and img_car.shape[2] == 3:
    b_channel, g_channel, r_channel = cv2.split(img_car)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 
    img_car = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

while True:
    success, img = cap.read() # đọc frame từ webcam
    if not success or img is None:
        break
    img = cvzone.overlayPNG(img, img_car, (0, 0))
    if img.shape[:2] != mask.shape[:2]:
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    else:
        mask_resized = mask
    imgRegion = cv2.bitwise_and(img, mask_resized) # áp dụng mask lên frame
    results = model(imgRegion, stream=True) # dự đoán object trong frame, stream=True để đọc frame liên tục
    # Kết quả dự đoán được lấy ra từ imgRegion có mask che
    # Những box nào không nằm trong mask sẽ không được dự đoán
    # Còn kết quả dự đoán sẽ được vẽ lên img (ảnh gốc)
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
            arr_class = ["xe dap", "xe hoi", "xe may", "xe buyt", "xe tai"]
            if classnames_vi[cls] in arr_class and conf > 0.3:
                # Vẽ text lên box
                # cvzone.putTextRect(img, f"{classnames_vi[cls]} {conf}", (max(0, x1), max(35, y1)),
                #                 scale=0.6, thickness=1, offset=5) # max(0, x1), max(35, y1) để tránh vượt quá biên của frame
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))
            
            
    result_trackers = tracker.update(detections) # result_trackers là một mảng 2 chiều, mỗi hàng là một box, mỗi cột là: x1, y1, x2, y2, conf, id
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5) # vẽ đường giới hạn lên frame
    for result in result_trackers:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w , h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=5, t=2, rt=5, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1)),
                                scale=1, thickness=2, offset=5, colorT=(255, 255, 255))
        
        cx, cy = x1 + w // 2, y1 + h // 2 # tọa độ của tâm của box
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED) # vẽ điểm lên frame
        
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15: # nếu tâm của box nằm trong đường giới hạn
            if id not in total_id:
                total_id.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # đổi màu đường giới hạn khi có xe đi qua
    
    # cvzone.putTextRect(img, f"Count: {len(total_id)}", (50, 50), scale=2, thickness=2, offset=5, colorT=(0, 0, 255))
    cv2.putText(img, f"{len(total_id)}", (85,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(1)
