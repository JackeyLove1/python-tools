from flask import Flask, request, jsonify
from text2vec import SentenceModel

app = Flask(__name__)
model = SentenceModel('shibing624/text2vec-base-chinese-sentence')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    sentence = data['sentence']
    embeddings = model.encode(sentence)
    return jsonify(embeddings)

if __name__ == '__main__':
    app.run(port=5000, debug=True)