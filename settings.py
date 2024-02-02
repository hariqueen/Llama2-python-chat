import os
import dotenv

env_path = dotenv.find_dotenv()
dotenv.load_dotenv(env_path)

DB_SETTINGS = {
    "MYSQL": {
        'host': os.environ.get("MYSQL_HOST"),
        'db_name': os.environ.get("MYSQL_DB_1"),
        'user': os.environ.get("MYSQL_USER"),
        'password': os.environ.get("MYSQL_PASSWORD")
    }
}