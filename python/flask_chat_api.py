import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from api_util.chat_completion import chat_with_bot

app = Flask(__name__)
CORS(app)

@app.get("/")
def hello_world():
  return "<p>Hello, World!</p>"

@app.post("/chat/completion")
def chat_completion():
  # print(request.json)
  data = request.json
  assert data['user_input'] != '', 'user_input must not be empty'
  completion = chat_with_bot(
    user_input=data['user_input'],
    max_length=data.get('max_length'),
    temperature=data.get('temperature'),
    top_p=data.get('top_p'),
    debug=data.get('debug')
  )
  return jsonify(
    result=completion
  )