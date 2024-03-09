import os
import pyodbc
from sqlalchemy import create_engine
from langchain.utilities import SQLDatabase
import warnings

warnings.filterwarnings("ignore")

server = 'nwu-capstone-2024.database.windows.net'
database = 'capstone'
username = 'team4'
password = '{capstone#123}'
driver= '{ODBC Driver 17 for SQL Server}'
    
class SQLDatabaseManager:
    _instance = None
    _connection = None
    _cursor = None

    def __init__(self):
        if not SQLDatabaseManager._connection:
            self._create_connection()

    @classmethod
    def get_connection(cls):
        if not cls._instance:
            cls._instance = cls({})  
        return cls._connection

    def _create_connection(self):
        SQLDatabaseManager._connection = pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) 
        SQLDatabaseManager._cursor = SQLDatabaseManager._connection.cursor()

    def __enter__(self):
        # self.connection = mysql.connector.connect(
        #     host=self.host, 
        #     user=self.user, 
        #     password=self.password, 
        #     database=self.database
        # )
        # self.cursor = self.connection.cursor()
        # return self
        return SQLDatabaseManager._connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        SQLDatabaseManager._connection.close()

    def execute_query(self, query, params=None):
        SQLDatabaseManager._cursor.execute(query, params)
        return SQLDatabaseManager._cursor.fetchall()
