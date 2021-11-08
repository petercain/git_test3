import sqlite3

try:

    conn = sqlite3.connect("record (2).db", isolation_level=None)
    cur = conn.cursor()
    # sql = "INSERT INTO MEMBER VALUES(?,?,?)"
    # cur.execute(sql, (435, 345, 345, ))
    # sql = "INSERT INTO data_record VALUES(?,?,?,?,?,?,?)"
    # cur.execute(sql, ('진실은',435,345,345,'언제나','단 하나!', 'model1'))
    cur.execute("SELECT * FROM data_record")
    result = cur.fetchall()
    print(result[0][0])


except:
    print("오류")