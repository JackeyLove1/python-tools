import os
import unittest
from flask import current_app, Flask, Blueprint
from flask_sqlalchemy import SQLAlchemy
basedir = os.path.abspath(os.path.dirname(__file__))
def create_app():
    app = Flask(__name__)
    page = Blueprint("page", __name__, url_prefix="/page")
    @page.route("/hello", methods=['GET', "POST"])
    def Hello():
        return "Hello, World!"
    app.register_blueprint(page)
    return app
def create_db(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = \
        'sqlite:///' + os.path.join(basedir, 'data.sqlite')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db = SQLAlchemy(app)
    return db

class BasicCase(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.db = create_db(self.app)
        self.db.drop_all()
        self.db.create_all()

    def tearDown(self) -> None:
        self.db.session.remove()
        self.db.drop_all()
        self.app_context.pop()

    def test_app_exists(self):
        self.assertFalse(current_app is None)
    def test_app_is_testing(self):
        pass