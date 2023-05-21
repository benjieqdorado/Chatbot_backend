import sqlite3
from sqlite3 import Error
import pickle

class DatabaseHelper:

 def create_connection(self,db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


 def create_chatlog(self,conn, chatlog):

    sql = ''' INSERT INTO chatlogs(uuid,role,time,message,vector)
              VALUES(?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, chatlog)
    conn.commit()
    return cur.lastrowid


 def create_notes(self,conn, notes):

    sql = ''' INSERT INTO notes(uuid,notes,time,uuids,times,vector)
              VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, notes)
    conn.commit()
    return cur.lastrowid
 

 def fetch_all_chatlogs(self,conn):
  
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM chatlogs")

    rows = cur.fetchall()

    log_list = []
    for log in rows:
        log_dict = {
            'uuid': log[0],
            'role': log[1],
            'time': log[2],
            'message': log[3],
            'vector': pickle.loads(log[4])
        }
        log_list.append(log_dict)
    return log_list




