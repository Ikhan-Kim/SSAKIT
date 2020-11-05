import sqlite3

print(sqlite3.version)
print(sqlite3.sqlite_version)

# DB 생성 (오토 커밋)
conn = sqlite3.connect("test.db", isolation_level=None)

# 커서 획득
c = conn.cursor()

# 테이블 생성 (데이터 타입은 TEST,  NUMERIC, INTEGER, REAL, BLOB 등)
c.execute("CREATE TABLE IF NOT EXISTS table1 \
    (id integer PRIMARY KEY, name text, brithday test)")


# 데이터 삽입 방법 1
c.execute("INSERT INTO table1 \
    VALUES(1, 'LEE', '1987-00-00')")


# 데이터 불러오기
c.execute("SELECT * FROM table1")
print(c.fetchone())