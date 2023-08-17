import hashlib
import pickle
import sqlite3
import time
from sqlite3 import Connection, Cursor, OperationalError
from typing import Any, Optional, Sequence, Union

from loguru import logger


class DAO:
    def __init__(self, db_file: str = "cache.db") -> None:
        self.db_file = db_file
        self.conn: Connection = self.init_db
        self.cursor: Cursor = self.conn.cursor()
        self.create_table("cache")

    @property
    def init_db(self) -> Connection:
        try:
            return sqlite3.connect(self.db_file)
        except (OperationalError, Exception) as e:
            logger.error("Failed to connect to sqlite db")
            raise e

    def create_table(self, table_name: str):
        create_table = f"""CREATE TABLE IF NOT EXIST {table_name} (
            key TEXT PRIMARY KEY,
            value BLOB,
            expiration INTEGER
        )"""
        self.cursor.execute(create_table)

    def _execute_sql(
        self, sql_stmt: str, parameters: Optional[Union[dict, Sequence]] = None
    ):
        if parameters:
            self.cursor.execute(sql_stmt, parameters)
        else:
            self.cursor.execute(sql_stmt)

    def get_from_cache(self, key: str):
        # Check if the result exists in the cache and if it has expired
        current_time = int(time.time())
        self._execute_sql(
            "SELECT value FROM cache WHERE key = ? AND expiration > ?",
            (key, current_time),
        )
        result = self.cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        else:
            return None

    def store_in_cache(self, key: str, value: Any, expiration_time: Union[int, float]):
        # Store the result in the cache with an expiration time
        pickled_value = pickle.dumps(value)
        self._execute_sql(
            "INSERT OR REPLACE INTO cache (key, value, expiration) VALUES (?, ?, ?)",
            (key, pickled_value, expiration_time),
        )
        self.conn.commit()

    @staticmethod
    def get_cache_key():
        # Generate a unique cache key based on script arguments or inputs
        # In this example, we use a hash of the script filename
        script_filename = "your_script.py"  # Replace with your actual script filename
        key = hashlib.md5(script_filename.encode()).hexdigest()

        return key
