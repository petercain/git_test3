import cv2
import numpy as np
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# image = cv2.imread('rr.jpg')
image = cv2.imread("iu.jpg", cv2.IMREAD_GRAYSCALE) # 흑백 이미지로 로드
# image= cv2.resize(image, (300,300)) # 이미지 크기를 50x50 픽셀로 변경
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(image, 1.3, 5)
imgNum  = 0
for (x,y,w,h) in faces:
    cropped_image = image[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    # 이미지를 저장
    # cv2.imwrite("thumbnail" + str(imgNum) + ".png", cropped)
    imgNum += 1


# plt.imshow(image) # 이미지를 출력
plt.imshow(cropped_image, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()

# cv2.imwrite("./data/images/plane_new.jpg", image) # 이미지를 저장

