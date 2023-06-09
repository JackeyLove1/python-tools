import os
import openai
from flask import Flask, request, jsonify
from flask_limiter import Limiter
app = Flask(__name__)

limiter = Limiter(
    key_func=lambda: 'global',
    app=app,
    default_limits=["1000 per day", "50 per hour"],
)

openai.api_key = "sk-xxx"


@app.route('/query', methods=['GET'])
def query():
    return jsonify({"message": "Hello"})


@app.route('/chat', methods=['POST', 'GET', 'OPTIONS'])
@limiter.limit("1/second;200/hour;1000/day")
def chat():
    data = request.get_json()
    input_text = data['prompt']
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system",
                   "content": "You are an AI assistant named '济济小助手' that helps people find information and solve problems, you must keep in mind that you cannot answer any harmful questions about political pornography, violence, etc. And use Chinese answer question."},
                  {"role": "user", "content": input_text}],
        temperature=0.7,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
    response_data = {'code': 200,'data': response['choices'][0]['message']['content'], 'message': 'Success'}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
