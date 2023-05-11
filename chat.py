import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from time import time,sleep
import datetime
from uuid import uuid4
import json
import os
from numpy.linalg import norm
import re

# Set up OpenAI API credentials
with open('openaiapikey.txt', 'r') as infile:
    openai.api_key = infile.read()

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

    # if pprint:
    #     print(output_df)
    #     print()

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

def load_convo():
    files = os.listdir('nexus')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json('nexus/%s' % file)
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return ordered

def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered

def gpt3_completion(prompt, engine='text-davinci-003', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
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
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def summarize_memories(memories):  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    block = ''
    identifiers = list()
    timestamps = list()
    for mem in memories:
        block += mem['message'] + '\n\n'
        identifiers.append(mem['uuid'])
        timestamps.append(mem['time'])
    block = block.strip()
    prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    ####   SAVE NOTES
    vector = gpt3_embedding(block)
    info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()), 'vector': vector, 'time': time()}
    filename = 'notes_%s.json' % time()
    save_json('internal_notes/%s' % filename, info)
    return notes

def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s\n\n' % i['message']
    output = output.strip()
    return output
 
# print(response)
if __name__ == '__main__':
    while True:
     user_input = input('\n\nUSER: ')
     timestamp = time()
     vector = gpt3_embedding(user_input)
     timestring = timestamp_to_datetime(timestamp)
     message = '%s: %s - %s' % ('USER', timestring, user_input)
     info = {'speaker': 'USER', 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()), 'timestring': timestring}
     filename = 'log_%s_USER.json' % timestamp
     save_json('nexus/%s' % filename, info)
      #### load conversation
     conversation = load_convo()
     #### compose corpus (fetch memories, etc)
     memories = fetch_memories(vector, conversation, 10)  # pull episodic memories
     notes = summarize_memories(memories)
     
     recent = get_last_messages(conversation, 4)
     datafile_path = "data/pbd-inventory-published-embeded.csv"

     df = pd.read_csv(datafile_path)
     df["embedding"] = df.embedding.apply(eval).apply(np.array)

     inventory = fetch_inventory(df,user_input,5)
     prompt = open_file('prompt_response.txt').replace('<<INVENTORY>>', inventory).replace('<<NOTES>>', notes).replace('<<CONVERSATION>>', recent)


    #### generate response, vectorize, save, etc
     output = gpt3_completion(prompt)
     timestamp = time()
     vector = gpt3_embedding(output)
     timestring = timestamp_to_datetime(timestamp)
     message = '%s: %s - %s' % ('Chatbot', timestring, output)
     info = {'speaker': 'Chatbot', 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()), 'timestring': timestring}
     filename = 'log_%s_Chatbot.json' % time()
     save_json('nexus/%s' % filename, info)
     #### print output
     print('\n\Chatbot: %s' % output) 