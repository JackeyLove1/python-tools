from flask import url_for, Flask
from markupsafe import escape
from flask_sqlalchemy import SQLAlchemy
import
app = Flask(__name__)
db = SQLAlchemy(app)

@app.route('/user/<name>', methods=['GET'])
def user_page(name) :

    return f'User: {escape(name)}'

app.run(port=8000)