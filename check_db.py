import pymysql

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'jaw_rehab',
    'autocommit': True
}

conn = pymysql.connect(**DB_CONFIG)
with conn.cursor() as cursor:
    cursor.execute("DESCRIBE patients")
    for row in cursor.fetchall():
        print(row)
conn.close()
