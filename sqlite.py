
import sqlite3
conn = sqlite3.connect("membershipDB.db", isolation_level=None)
cur = conn.cursor()

conn.execute("INSERT INTO order_history(id,time,menu,cost,PM) VALUES(?,?)",
                      (self.label_9.text(),now))

a = cur.fetchall()
for i in range(len(a)):
    self.textBrowser_3.append("{}\n{}\n{}\n{}\n" .format(a[i][1],a[i][2],a[i][3],a[i][4]))


        cur.execute("SELECT pw FROM member WHERE id = '{}'".format(id))
        result = cur.fetchall()
        for row in result:
            if row[0] == pw:
                self.window1.label_9.setText(id)
                self.window1.lineEdit_2.setText(pw)
                self.window1.show()
                self.close()
            else:
                pyautogui.alert("아이디나 비밀번호를 다시 확인해주세요")


        id_findall = re.findall(' ', id)
        cur.execute("SELECT id FROM member WHERE id = '{}'".format(id))
        result = cur.fetchall()
        if not result:
