from keras.models import load_model
import tensorflow.keras
import numpy as np
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic, QtGui
import sys
import pyautogui
import sqlite3

try:
    conn = sqlite3.connect("record (2).db", isolation_level=None)
    cur = conn.cursor()
    cur2 = conn.cursor()

    # global model
    # global face_cascade
    model = load_model('keras_model.h5')

except:
    print("디비 연결 실패")

ui = uic.loadUiType("face.ui")[0]

class Main(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton_backup.clicked.connect(self.upload_file)
        self.pushButton_test.clicked.connect(self.check_image)
        self.pushButton_record.clicked.connect(self.go_record)
        self.pushButton_back.clicked.connect(self.go_back)

        # ui  이미지 픽스
        self.pushButton_back.setIcon(QtGui.QIcon("back.PNG"))
        self.pushButton_record.setIcon(QtGui.QIcon("record.PNG"))
        self.pushButton_backup.setIcon(QtGui.QIcon("file.PNG"))
        self.pushButton_test.setIcon(QtGui.QIcon("test.PNG"))
        self.label_2.setPixmap(QtGui.QPixmap("percent.PNG"))
        self.label_3.setPixmap(QtGui.QPixmap("nemo.PNG"))
        self.label.setPixmap(QtGui.QPixmap("menu.PNG"))

        self.pushButton_back.setDisabled(True)

    def upload_file(self):
        filter = 'Image(*.png *.jpg *.jpeg *.PNG bmp ) (.png *.jpg *.jpeg *bmp *.PNG)'
        self.image_path = QFileDialog.getOpenFileName(self, '파일 선택', filter=filter)
        self.image_path = self.image_path[0]
        image = QPixmap()
        image.load(self.image_path)  #이미지
        self.img.setPixmap(image)  #img라벨에 삽입

    def check_image(self):
        try:

            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            img = cv2.imread(self.image_path)  # 경로
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(224, 224))




            faces = face_cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=1,
                                         minSize=(10,10))
            # faces = face_cascade.detectMultiScale(img, 1.3, 5)

            if len(faces) > 0:

               #이미지 자르기
               for (x, y, w, h) in faces:
                        cropped_image = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
                        cv2.imwrite("thumbnail" + ".png", cropped_image)

               # 이미지 분석
               np.set_printoptions(suppress=True)
               # model = tensorflow.keras.models.load_model('keras_model.h5')

               data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

               img2 = cv2.resize(cropped_image, (224, 224))

               image_array = np.asarray(img2)
               normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

               data[0] = normalized_image_array
               prediction = model.predict(data)

               print(prediction)
               for i in prediction:
                   self.state = ""
                   self.smile = int(i[0] * 100)
                   self.sad = int(i[1] * 100)
                   self.angry= int(i[2] * 100)
                   self.label_smile.setText(str(self.smile) + "%")
                   self.label_angry.setText(str(self.angry) + "%")
                   self.label_sad.setText(str(self.sad) + "%")
                   if i[0] > 0.7:
                       self.label_state.setText("smile")
                       self.state = "smile"
                       print("웃음 : ", i[0])
                       print("슬픔 : ", i[1])
                       print("화남 : ", i[2])
                       pyautogui.alert('smile 얼굴입니다.')
                   elif i[1] > 0.7:
                       self.label_state.setText("sad")
                       self.state = "sad"
                       print("웃음 : ", i[0])
                       print("슬픔 : ", i[1])
                       print("화남 : ", i[2])
                       pyautogui.alert('sad 얼굴입니다')
                   elif i[2] > 0.7:
                       self.label_state.setText("angry")
                       self.state = "angry"
                       print("웃음 : ", i[0])
                       print("슬픔 : ", i[1])
                       print("화남 : ", i[2])
                       pyautogui.alert('angry 얼굴입니다')
                   else:
                       self.state = "?"
                       pyautogui.alert('표정을 알 수 없습니다')
                       self.label_state.setText("???")
               # (image_path, smile_percent, angry_percent, sad_percent, AI_choice, user_check)
               self.confirm = pyautogui.confirm('맞습니까?', buttons=['OK', 'NG'])
               try:
                   sql = "INSERT INTO data_record VALUES(?,?,?,?,?,?,?)"
                   cur.execute(sql, (self.image_path, self.smile, self.angry, self.sad, self.state, self.confirm, "model1"))
                   print("데이터 입력에 성공하였습니다.")
               except:
                   print("데이터 입력에 실패하였습니다.")

            else:
                pyautogui.alert('얼굴을 인식할 수 없습니다.')
        except:
            pyautogui.alert('다른 사진을 사용해주세요ㅠㅠ')



    def go_record(self):
        index = self.stackedWidget.currentIndex()
        self.stackedWidget.setCurrentIndex(index+1)

        cur.execute("SELECT * FROM data_record")
        result = cur.fetchall()
        count = len(result)
        self.tableWidget.setRowCount(count)
        self.tableWidget.setColumnCount(7)
        for y in range(7):
            for x in range(count):
                # 테이블의 각 셀에 값 입력
                self.tableWidget.setItem(x, y, QTableWidgetItem(str(result[x][y])))

        # 신뢰도 계산
        cur.execute("SELECT count(*) FROM data_record")
        result2 = cur.fetchall()
        count = result2[0][0] # 총데이터개수
        cur.execute("SELECT count(*) FROM data_record WHERE user_check ='OK' ")
        result3 = cur.fetchall()
        count_OK = result3[0][0]  # OK데이터개수
        reliability = round((count_OK/count)*100,1)
        self.label_reliability.setText(str(reliability)+"%")


        self.pushButton_test.setDisabled(True)
        self.pushButton_backup.setDisabled(True)
        self.pushButton_record.setDisabled(True)
        self.pushButton_back.setEnabled(True)

    def go_back(self):
        index = self.stackedWidget.currentIndex()
        self.stackedWidget.setCurrentIndex(index-1)

        self.pushButton_back.setDisabled(True)
        self.pushButton_test.setEnabled(True)
        self.pushButton_backup.setEnabled(True)
        self.pushButton_record.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    start = Main()
    start.show()
    app.exec_()