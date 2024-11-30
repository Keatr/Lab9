import cv2
import numpy as np

# Tải mô hình Caffe
model = "res10_300x300_ssd_iter_140000.caffemodel"
config = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config, model)

def detect_faces(image_path):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    
    # Xử lý ảnh: Thay đổi kích thước và trừ đi giá trị trung bình
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # Đưa blob qua mạng để nhận diện
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Lọc các phát hiện yếu
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Vẽ khung 
            text = f"{confidence:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    #anh đầu ra
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_faces("test.png")
