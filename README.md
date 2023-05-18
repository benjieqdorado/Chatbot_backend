## chatbot backend
This openai application will answer question based on our gun database. It uses embedding to match the users questions with our list of posible question and gun descriptions

## REQUIREMENTS:
Python 3.x

pip install -U Flask
pip install openai
pip install pandas
pip install numpy
pip install pprint
pip install Flask-RESTful
pip install -U flask-cors
pip install python-dotenv
pip install vdblite

## FIRST TIME USAGE:
install python prerequisites
create .env file in your root directory (same directory as your main.py) and copy the content of .env.back.
copy the content of the .env.back file and update the OPENAI_API_KEY.
run the code by typing python main.py in your cmd while inside the project directory.
COMMAND (project_root/commands/) files for generating the data:
prepare_embedding_data.py
this will create a new csv file based on the gun_data that we have. We must run this first to format the csv for the embedding.
create_embedding_for_gun_data.py
will create another column in our gun_data_embedding.csv to add the generated vector representation of the text.

## API USAGE:
We used python FLASK for creating the api. You need to run our main.py file using this command python main.py first to be able to use the api's.
## API DOCUMENTATION:
GET: /chatgpt/question

get all messages