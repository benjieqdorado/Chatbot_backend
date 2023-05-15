import sqlite3

conn = sqlite3.connect('chatbot.db')

cur = conn.cursor()

cur.execute('''CREATE TABLE chat_log
               (id INT PRIMARY KEY NOT NULL,
                role TEXT NOT NULL,
                timestamp INT NOT NULL,
                message TEXT NOT NULL,
                vector TEXT NOT NULL
                )''')


cur.execute('''CREATE TABLE notes
               (id INT PRIMARY KEY NOT NULL,
                notes TEXT NOT NULL,
                time INT NOT NULL,
                times TEXT NOT NULL,
                chatlog_ids INT NOT NULL,
                vector TEXT NOT NUll
                )''')


# cur.execute("SELECT * FROM chat_log")

# rows = cur.fetchall()
# print(rows)
# for row in rows:
#     print(row)
conn.commit()
conn.close()