import hashlib
import pickle
import sqlite3
import time
from sqlite3 import Connection, Cursor, OperationalError
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from loguru import logger


class DAO:
    def __init__(
        self,
        db_file: str = "cache.db",
        table_name: str = "cache",
        expired_time: int = 3600,
    ) -> None:
        self.db_file = db_file
        self.expired_time = expired_time

        # with self.init_db() as connection:
        self.conn: Connection = next(self.init_db())
        self.cursor: Cursor = self.conn.cursor()

        self.table_name = table_name
        self.create_table(table_name)

    def init_db(self) -> Connection:
        try:
            yield sqlite3.connect(self.db_file)
        except (OperationalError, Exception) as e:
            logger.error("Failed to connect to sqlite db")
            raise e

    def _execute_sql(
        self, sql_stmt: str, parameters: Optional[Union[dict, Sequence]] = None
    ):
        if parameters:
            self.cursor.execute(sql_stmt, parameters)
        else:
            self.cursor.execute(sql_stmt)

    def create_table(self, table_name: str):
        create_table = f"""CREATE TABLE IF NOT EXISTS {table_name} (
            key TEXT PRIMARY KEY,
            value BLOB,
            expiration INTEGER
        )"""
        self._execute_sql(create_table)

    def get_from_cache(self, db_key: str):
        # Check if the result exists in the cache and if it has expired
        current_time = int(time.time())
        self._execute_sql(
            f"SELECT value FROM {self.table_name} WHERE key = ? AND expiration > ?",
            (db_key, current_time),
        )
        result = self.cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        else:
            return None

    def store_in_cache(
        self, db_key: str, value: Any, expiration_time: Union[int, float]
    ):
        # Store the result in the cache with an expiration time
        pickled_value = pickle.dumps(value)
        self._execute_sql(
            f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
            (db_key, pickled_value, expiration_time),
        )
        self.conn.commit()

    @staticmethod
    def get_cache_key(key: str):
        # Generate a unique cache key based on script arguments or inputs
        db_key = hashlib.md5(key.encode()).hexdigest()

        return db_key

    def get_result(
        self,
        key: str,
        nocache_fn: Callable[[Union[str, list]], Tuple[str, bool]],
        param: Union[str, list],
    ) -> str:
        cache_key = self.get_cache_key(key)
        cache_res = self.get_from_cache(cache_key)

        if cache_res:
            logger.debug("\nResult (from cache)")
            return (cache_res,)
        else:
            cost_time_res, bool_r = nocache_fn(param)
            if bool_r:
                expiration_time = int(time.time()) + self.expired_time
                self.store_in_cache(cache_key, cost_time_res, expiration_time)
                logger.debug("\nOrigin result")

                return (cost_time_res,)
            else:
                raise ValueError(f"The result if wrong. Command: {param}")
