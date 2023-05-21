import sqlite3
from sqlite3 import Error
import os


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)



def main(db_folder,db_name):
    database = os.path.join(os.getcwd(), db_folder, db_name)

    sql_create_chatlogs_table = """ CREATE TABLE IF NOT EXISTS chatlogs (
                                        uuid text PRIMARY KEY,
                                        role text NOT NULL,
                                        time integer NOT NULL,
                                        message text NOT NULL,
                                        vector blob NOT NULL
                                    ); """

    sql_create_notes_table = """CREATE TABLE IF NOT EXISTS notes (
                                    uuid text PRIMARY KEY,
                                    notes text NOT NULL,
                                    time integer NOT NULL,
                                    uuids text NOT NULL,
                                    times integer NOT NULL,
                                    vector blob NOT NULL
                                );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create chatlogs table
        create_table(conn, sql_create_chatlogs_table)

        # create notes table
        create_table(conn, sql_create_notes_table)
    else:
        print("Error! cannot create the database connection.")


if __name__ == '__main__':
    db_folder = "db"
    db_name = "chatbot.db"
    main(db_folder,db_name)