import time
import os
from random import randint
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import datetime
from faker import Faker
from flask_migrate import upgrade

basedir = os.getcwd()
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = \
    'sqlite:///' + os.path.join(basedir, 'test.sqlite')
db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    password = db.Column(db.String(128))
    about = db.Column(db.Text())
    create_at = db.Column(db.DateTime(), default=datetime.datetime.utcnow())


fake = Faker()
Faker.seed(int(time.time()))
print(fake.email())
print(fake.user_name())
print(fake.name())
print(fake.time())
print(fake.company())
print(fake.json())
print(fake.texts())
print(fake.address())
print(fake.city())

with app.app_context():
    db.create_all()
def Insert_Users(count=100):
    Faker.seed(int(time.time()))
    fake = Faker()
    for _ in range(count):
        u = User(email=fake.email(),
                 username=fake.user_name(),
                 password="password",
                 about=fake.text())
        db.session.add(u)
    try:
        db.session.commit()
    except Exception as e:
        print("Faield to add user, Error:", str(e))
        db.session.rollback()
