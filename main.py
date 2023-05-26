from flask import Flask, request, jsonify, Response
from flask_restful import Api, Resource
import pprint
from flask_cors import CORS
from shared.constant import Constant
from services.ChatGPT import ChatBot


app = Flask(__name__)
cors = CORS(app)  # all api calls below will be CORS ready
app.config['CORS_HEADERS'] = 'Content-Type'

CONSTANTS = Constant
pp = pprint.PrettyPrinter(indent=4)

chatbot = ChatBot()
###############################
### Chat GPT api request    ###
###############################


@app.route(CONSTANTS.URLS['CHATGPT_QUESTION'], methods=['GET'])
def get():
    return chatbot.get_chat_messages()


@app.route(CONSTANTS.URLS['CHATGPT_QUESTION'], methods=['POST'])
def process_question():

    response = chatbot.ask_chatgpt_question(request.json['question'])
    return Response(response, content_type='text/event-stream')




###############################
### END chat gpt request    ###
###############################


if __name__ == '__main__':
    app.run()
