import mysql.connector

# DB 연결을 위한 클래스
class DBconnector:

    # DB 연결 매개변수
    def __init__(self, host, db_name, user, password):
        self.conn_params = dict(
            host = host,
            database = db_name,
            user = user,
            password = password
        )

        # MySQL 연결
        self.connect = self.mysql_connect()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()

    def mysql_connect(self):
        self.conn = mysql.connector.connect(**self.conn_params)