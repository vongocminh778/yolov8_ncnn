import cv2
import os

# Kiểm tra nếu thư mục 'capture_imgs' không tồn tại, tạo mới nó
if not os.path.exists('capture_imgs'):
    os.makedirs('capture_imgs')

# Khởi tạo biến đếm cho tên ảnh
img_counter = 0

# Khởi tạo biến cho camera
cap = cv2.VideoCapture(0)

# Thiết lập kích thước cho khung hình
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    # Hiển thị khung hình trong cửa sổ
    cv2.imshow('Press S to Save Image', frame)

    # Chờ bấm phím
    key = cv2.waitKey(1)

    # Nếu bấm phím 's' (Save)
    if key == ord('s'):
        # Tạo tên file ảnh
        img_name = f'capture_imgs/img_{img_counter}.png'

        # Lưu ảnh vào thư mục 'capture_imgs'
        cv2.imwrite(img_name, frame)
        print(f'{img_name} đã được lưu.')

        # Tăng biến đếm
        img_counter += 1

    # Nếu bấm phím 'q' (Quit)
    elif key == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()