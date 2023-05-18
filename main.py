from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pprint
from flask_cors import CORS
from shared.constant import Constant
from services.ChatGPT import ask_chatgpt_question,get_chat_messages
import datetime
import time

app = Flask(__name__)
cors = CORS(app)  # all api calls below will be CORS ready
app.config['CORS_HEADERS'] = 'Content-Type'

CONSTANTS = Constant
pp = pprint.PrettyPrinter(indent=4)

###############################
### Chat GPT api request    ###
###############################


@app.route(CONSTANTS.URLS['CHATGPT_QUESTION'], methods=['GET'])
def get():
    return get_chat_messages()


@app.route(CONSTANTS.URLS['CHATGPT_QUESTION'], methods=['POST'])
def askQuestion():
    start_time = time.time()
    response = ask_chatgpt_question(request.json['question'])
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    # Format the elapsed time as a string in a human-readable format
    time_str = datetime.timedelta(seconds=int(round(time_elapsed)))
    
    return jsonify({'result': response, 'time_response': str(time_str)})
  


###############################
### END chat gpt request    ###
###############################


if __name__ == '__main__':
    app.run()
