import openai
import pandas as pd
import numpy as np
import os
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
from time import time,sleep
import datetime
from uuid import uuid4
import json
from numpy.linalg import norm
import pickle
from datetime import datetime
import vdblite
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


datafile_path = "data/pbd-inventory-published-embeded.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
          
def fetch_inventory(df, search_term, n=3, pprint=True):

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
                             "images", "product_category", "product_brand","product_caliber","product_link"])


    output_string = output_df.to_string(index=False)

    return output_string

def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  # return cosine similarity

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)
    
def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

def load_convo(cur):
    cur.execute("SELECT id, role, timestamp, message, vector FROM chat_log")
    rows = cur.fetchall()
    log_list = []
    for log in rows:
        log_dict = {
            'id': log[0],
            'role': log[1],
            'timestamp': log[2],
            'message': log[3],
            'vector': pickle.loads(log[4])
        }
        log_list.append(log_dict)
    return log_list


async def fetch_memories(vector, logs, count):
    scores = []

    for log in logs:

        if vector == log['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(log['vector'], vector)
        log['score'] = score
        scores.append(log)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered
    

def gpt3_completion(prompt, engine='text-davinci-003', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['Customer:', 'Chatbot:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
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
                stop=stop)
            text = response['choices'][0]['text'].strip()
            # text = re.sub('[\r\n]+', '\n', text)
            # text = re.sub('[\t ]+', ' ', text)
            # filename = '%s_gpt3.txt' % time()
            # if not os.path.exists('gpt3_logs'):
            #     os.makedirs('gpt3_logs')
            # save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def summarize_memories(memories):  
   # Summarizes a block of memories into one payload.

    # Sort memories chronologically
    memories = sorted(memories, key=lambda d: d['timestamp'])

    # Collect message content, identifiers, and timestamps
    message_block = ''
    identifiers = []
    timestamps = []
    for memory in memories:
        message_block += memory['message'] + '\n\n'
        identifiers.append(memory['id'])
        timestamps.append(memory['timestamp'])


    # Remove trailing whitespace
    message_block = message_block.strip()

    # Use the message block to generate notes using GPT-3
    if message_block == '':
        prompt = ''
    else:
        with open('prompt_notes.txt', 'r') as f:
            prompt = f.read().replace('<<INPUT>>', message_block)
    
    notes = gpt3_completion(prompt)

    # Save notes and metadata to database
    # vector = gpt3_embedding(message_block)
    # notes_id = str(uuid4())
    # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # timestamp_str = ','.join(map(str, timestamps))
    # chatlogs_ids = ','.join(map(str, identifiers))
    # vector_blob = pickle.dumps(vector)
    # conn = sqlite3.connect('chatbot.db')
    # cur = conn.cursor()
    # cur.execute("INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?)",
    #             (notes_id, notes, timestamp, timestamp_str, chatlogs_ids, vector_blob))
    # conn.commit()
    # conn.close()

    return notes

def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s: %s\n\n' % (i['role'], i['message'])  # Added role before the message
    output = output.strip()
    return output

 

vdb = vdblite.Vdb()
# vdb.load('my_data.vdb') or 

def ask_chatgpt_question(question):
    # Insert customer's message into chat_log table
    timestamp = time()
    vector = gpt3_embedding(question)
    info = {'id': str(uuid4()), 'role': 'user', 'timestamp': timestamp, 'message':question ,'vector':vector}
    vdb.add(info)
    vdb.details()
    vdb.save('my_data.vdb')
    conversation = vdb.search(vector, 'vector', 10)
    notes = summarize_memories(conversation)
    # Fetch recent messages and inventory
    recent = get_last_messages(vdb.fetchAll(), 5)
    df = pd.read_csv("data/pbd-inventory-published-embeded.csv")
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    inventory = fetch_inventory(df, question, 3)
   
    recent = question
    # Generate response
    prompt = open_file('prompt_response.txt').replace('<<INVENTORY>>', inventory).replace('<<NOTES>>', notes).replace('<<CONVERSATION>>', recent)
    output = gpt3_completion(prompt)
    
    # Insert chatbot's message into chat_log table
    timestamp = time()
    vector = gpt3_embedding(output)
    info = {'id': str(uuid4()), 'role': 'assistant', 'timestamp': timestamp, 'message':output ,'vector':vector}
    vdb.add(info)
    vdb.details()
    vdb.save('my_data.vdb')
   # Return output
    return output


def get_chat_messages():
    vdb = vdblite.Vdb()
    vdb.load('my_data.vdb')
    result = vdb.fetchAll()
    
    return result or []






