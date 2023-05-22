import openai
import pandas as pd
import numpy as np
import os
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
from time import time, sleep
from uuid import uuid4
from numpy.linalg import norm
import pickle
from datetime import datetime
from utils.database_helper import DatabaseHelper
import pickle
import re
from utils.file_helper import FileHelper

load_dotenv()



openai.api_key = os.getenv('OPENAI_API_KEY')
datafile_path = "data/pbd-inventory-published-embeded.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

class ChatBot:


 def __init__(self):
        self.create = DatabaseHelper()
        self.file = FileHelper()
 
 def fetch_inventory(self,df, search_term, n=3, pprint=True):

    search_embedding = get_embedding(
        search_term,
        engine="text-embedding-ada-002"
    )

    df["similarity"] = df.embedding.apply(
        lambda x: cosine_similarity(x, search_embedding))
    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
    )

    rows = []
    for _, row in results.iterrows():
        product_name = row["product_name"]
        stock_status = row["stock_status"]
        regular_price = row["regular_price"]
        images = row["images"]
        product_category = row["product_category"]
        product_brand = row["product_brand"]
        product_caliber = row["product_caliber"]
        product_link = row["product_link"]

        rows.append({
            "product_name": product_name,
            "stock_status": stock_status,
            "regular_price": regular_price,
            "images": images,
            "product_category": product_category,
            "product_brand": product_brand,
            "product_caliber": product_caliber,
            "product_link": product_link,
        })

    output_df = pd.DataFrame(rows, columns=["product_name", "stock_status", "regular_price",
                             "images", "product_category", "product_brand", "product_caliber", "product_link"])

    output_string = output_df.to_string(index=False)

    return output_string


 def gpt3_embedding(self,content, model='text-embedding-ada-002'):
    content = str(content)
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content,model=model,)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


 def fetch_memories(self,vector, logs, count):
    scores = []

    for log in logs:
        if vector == log['vector']:
            # skip this one because it is the same message
            continue
        score = cosine_similarity(log['vector'], vector)
        log['score'] = score
        scores.append(log)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


 def gpt3_completion(self,prompt, engine='text-davinci-003', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['User:', 'Assistant:'],stream = False):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop,
                stream=stream
                )
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            self.file.save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


 def summarize_memories(self,conn,memories):
   # Summarizes a block of memories into one payload.

    # Sort memories chronologically
    memories = sorted(memories, key=lambda d: d['time'])

    # Collect message content, identifiers, and timestamps
    message_block = ''
    identifiers = []
    timestamps = []
    for memory in memories:
        message_block += memory['role'].upper() + ': ' + memory['message'] + '\n\n'
        identifiers.append(memory['uuid'])
        timestamps.append(memory['time'])

    # Remove trailing whitespace
    message_block = message_block.strip()

    # Use the message block to generate notes using GPT-3
    if message_block == '':
        prompt = ''
    else:
        with open('prompt_notes.txt', 'r') as f:
            prompt = f.read().replace('<<INPUT>>', message_block)

    notes = self.gpt3_completion(prompt)
    

    # Save notes and metadata to database
    vector = self.gpt3_embedding(message_block)
    serialized_vector = pickle.dumps(vector)
    timestamp_str = ','.join(map(str, timestamps))
    uuids = ','.join(map(str, identifiers))
    uuid = str(uuid4())
    timestamp = time()
    note_data = (uuid, notes, timestamp, uuids, timestamp_str,serialized_vector)
    self.create.create_notes(conn, note_data)

    return notes


 def get_last_messages(self,conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        # Added role before the message
        output += '%s: %s\n\n' % (i['role'].upper(), i['message'])
    output = output.strip()
    return output


 def ask_chatgpt_question(self,question):

    db_folder = "db"
    db_name = "chatbot.db"
    database = os.path.join(os.getcwd(), db_folder, db_name)

    conn = self.create.create_connection(database)
    # Insert customer's message into chat_log table
    timestamp = time()
    vector = self.gpt3_embedding(question)
    serialized_vector = pickle.dumps(vector)
    uuid = str(uuid4())
    role = 'user'
    chatlog = (uuid, role, timestamp, question, serialized_vector)
    self.create.create_chatlog(conn, chatlog)

    conversation = self.create.fetch_all_chatlogs(conn)

    memories = self.fetch_memories(vector, conversation, 10)

    notes = self.summarize_memories(conn,memories)
    
    # Fetch recent messages and inventory
    recent = self.get_last_messages(conversation, 4)
    df = pd.read_csv("data/pbd-inventory-published-embeded.csv")
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    inventory = self.fetch_inventory(df, question, 3)

    # Generate response
    prompt = self.file.open_file('prompt_response.txt').replace('<<INVENTORY>>', inventory).replace('<<NOTES>>', notes).replace('<<CONVERSATION>>', recent)
    output = self.gpt3_completion(prompt)
   
    # Insert chatbot's answer into chat_log table
    timestamp = time()
    vector = self.gpt3_embedding(question)
    serialized_vector = pickle.dumps(vector)
    uuid = str(uuid4())
    role = 'assistant'
    chatlog = (uuid, role, timestamp, output, serialized_vector)
    self.create.create_chatlog(conn, chatlog)
   # Return output
    return output


 def get_chat_messages(self):

    db_folder = "db"
    db_name = "chatbot.db"
    database = os.path.join(os.getcwd(), db_folder, db_name)

    conn = self.create.create_connection(database)
    curr = conn.cursor()

    curr.execute(f"SELECT uuid, role, time, message FROM chatlogs")

    rows = curr.fetchall()

    log_list = []
    for log in rows:
        log_dict = {
            'uuid': log[0],
            'role': log[1],
            'time': log[2],
            'message': log[3],
        }
        log_list.append(log_dict)
    
    return log_list
