from flask import Flask
from flask import Blueprint

app = Flask(__name__)
my_blueprint = Blueprint('my_blueprint', __name__, url_prefix='/myapp')

@app.route('/', methods=['GET', 'POST'])
def hello():
    return "Hello, App!"
@my_blueprint.route('/', methods=['GET', 'POST'])
def index():
    return 'Hello, world!'

@my_blueprint.route('/about', methods=['GET', 'POST'])
def about():
    return 'About us'

app.register_blueprint(my_blueprint)


if __name__ == '__main__':
    app.run()
