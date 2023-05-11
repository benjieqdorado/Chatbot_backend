from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pprint
from flask_cors import CORS
from shared.constant import Constant
from services.ChatGPT import ask_chatgpt_question

app = Flask(__name__)
cors = CORS(app)  # all api calls below will be CORS ready
app.config['CORS_HEADERS'] = 'Content-Type'

CONSTANTS = Constant
pp = pprint.PrettyPrinter(indent=4)

###############################
### Chat GPT api request    ###
###############################


# @app.route(CONSTANTS.URLS['CHATGPT_QUESTION'], methods=['GET'])
# def get():
#     return chatgpt.get_chat_messages()


@app.route(CONSTANTS.URLS['CHATGPT_QUESTION'], methods=['POST'])
def askQuestion():
    response = ask_chatgpt_question(request.json['question'])
    return jsonify({'result': response})



###############################
### END chat gpt request    ###
###############################


if __name__ == '__main__':
    app.run()
