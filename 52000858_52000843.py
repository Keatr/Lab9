import cv2
import numpy as np
import math

# TASK 1
# Hàm để phát hiện biển báo giao thông hình tròn dựa trên màu sắc
def detect_traffic_signs(frame):
    # Chuyển đổi sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Định nghĩa khoảng màu để phát hiện biển báo đỏ
    lower_red1 = np.array([0, 100, 50])  
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50]) 
    upper_red2 = np.array([180, 255, 255])

    # Định nghĩa khoảng màu để phát hiện biển báo xanh
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Tạo mặt nạ cho màu đỏ
    mask1_red = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2_red = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask1_red | mask2_red

    # Tạo mặt nạ cho màu xanh
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Kết hợp mặt nạ: chỉ giữ lại màu đỏ và xanh
    mask = mask_red | mask_blue

    # Chỉ giữ lại nửa trên khung hình
    height, width = frame.shape[:2]
    mask[height // 2:, :] = 0  # Đặt mặt nạ của nửa dưới bằng 0
    # Chỉ giữ lại khoảng từ 1/3 đến 2/3 của khung hình
    mask[:, :width // 3] = 0  # Đặt mặt nạ bên trái bằng 0
    mask[:, 2 * width // 3:] = 0  # Đặt mặt nạ bên phải bằng 0

    # Tìm các đường viền trong mặt nạ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vẽ ô vuông quanh các biển báo hình tròn phát hiện được
    for contour in contours:
        # Tính diện tích và độ tròn của đường viền
        area = cv2.contourArea(contour)
        if area > 500:  # Chỉ giữ lại các đường viền có diện tích lớn
            # Tính chu vi
            perimeter = cv2.arcLength(contour, True)
            # Tính số đỉnh gần đúng
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Kiểm tra xem đường viền có phải là hình tròn
            if len(approx) >= 8:  # Thay đổi số lượng đỉnh tùy thuộc vào yêu cầu
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Đọc video từ tệp
cap = cv2.VideoCapture('task1.mp4')

# Lấy thông tin về chiều rộng và chiều cao của video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Khởi tạo VideoWriter để lưu video đầu ra
out = cv2.VideoWriter('52000858_52000843.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện biển báo giao thông hình tròn
    detect_traffic_signs(frame)

    # Chèn dãy số vào góc trên cùng
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "52000858_52000843"
    cv2.putText(frame, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Ghi khung hình đã xử lý vào video đầu ra
    out.write(frame)

    # Hiển thị kết quả
    '''cv2.imshow('52000858_52000843', frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break'''

cap.release()
out.release()  
cv2.destroyAllWindows()


# TASK 2

#Định nghĩa phương thức làm nét ảnh
def sharpen(img):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return image_sharp

def digits_detect():
    init_img = cv2.imread(r"input.png")                     #Lấy ảnh input 
    result_img, img_copy = init_img.copy(), init_img.copy() #Copy ảnh gốc và lưu vào 2 biến (1 để lưu kq; xử lý ảnh chia làm 2 phần, 1 dùng ảnh gốc và 1 dùng bản copy)
    
    
    # XỬ LÝ NHỮNG CHỖ BỊ NHIỄU  CỦA ẢNH     
    img = sharpen(init_img)                                                             #Làm nét ảnh gốc
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                    #Chuyển ảnh xám
    img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)                                 #Sử dụng bộ lọc Gaussian để làm mờ ảnh xám 
    _, binary = cv2.threshold(img_blurred, 7, 255, cv2.THRESH_BINARY_INV)               #Sử dụng ngưỡng hóa (thresholding) để tạo ra một ảnh nhị phân binary, trong đó các pixel có giá trị dưới 7 sẽ được gán giá trị 0 (đen) và các pixel có giá trị từ 7 trở lên sẽ được gán giá trị 255 (trắng). Kết quả ảnh được đảo ngược (inverted) để các vùng quan tâm trở thành vùng trắng trên nền đen.
    contours, _ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    #Sử dụng để tìm các đường viền trên ảnh.
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)                                    #x, y = tọa độ điểm; w,h = độ dài các cạnh hình chữ nhật
        s = cv2.contourArea(c)                                              #Tính diện tích của c trong countours
        if s > 300:                                                         #Xét điều kiện nếu diện tích s(c) > 300 thì duyệt       
            cv2.rectangle(result_img,(x,y), (x+w,y+h), (0,255,0), 2)        #Vẽ hình chữ nhật bằng các thông số lấy ra từ hàm boundingRect sau khi qua đk, với màu đỏ và dộ dày đường viền bằng 2
    
    
    # XỬ LÝ NHỮNG CHỖ KHÔNG BỊ NHIỄU 
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)                            #Như trên
    _, binary = cv2.threshold(img_gray, 82, 255, cv2.THRESH_BINARY_INV)              #Như trên
    contours, _ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Như trên
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)                                #Như trên        
        d = math.sqrt(w^2 + h^2)                                        #Tính khoảng cách đường chéo hình chữ nhật
        if (cv2.contourArea(c)) > 30 and d <11:                         #Xét điều kiện nếu diện tích của c > 30 VÀ đường chéo hcn d < 11 thì duyệt
            cv2.rectangle(result_img,(x,y), (x+w,y+h), (0,255,0), 2)    #Như trên
    

    cv2.imwrite('52000858_52000843.jpg', result_img)  #Xuất ảnh output 


    pass

digits_detect()
