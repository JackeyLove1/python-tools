# pip3 install pymysql
# pip install SQLAlchemy
# pip install flask-sqlalchemy
# pip install -U Flask-SQLAlchemy
'''
Integer int32
SmallInteger int16
BigInteger bigInteger
Float float
Numeric decimal.decimal
String str
Text str
Unicode unicode
Boolean bool
Date datatime

primary_key 主键
unique  唯一
index 索引
nullable 是否允许使用空键
default 列的默认值
'''
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
basedir = os.path.abspath(os.path.dirname(__file__))
# basedir = os.getcwd()
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    def __repr__(self):
        return '<Role:{}>'.format(self.name)

class User(db.Model):
    __tabelname__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, index=True)
    def __repr__(self):
        return '<User:{}>'.format(self.username)

with app.app_context():
    db.drop_all()
    db.create_all()
    admin_role = Role(name="Admin")
    mod_role = Role(name="Moderator")
    user_role = Role(name="User")
    print(admin_role.id)
    db.session.add(admin_role)
    db.session.add_all([mod_role, user_role])
    db.session.commit()
    print(admin_role.id)
    print(Role.query.all())
