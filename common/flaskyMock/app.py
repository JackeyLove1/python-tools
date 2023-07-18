from flask import url_for, Flask
from markupsafe import escape
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
print(app.root_path)
# /Users/jacky/Desktop/python-tools/common/flaskyMock
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////' + os.path.join(app.root_path, 'data.db')
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20))


class Movie(db.Model):  # 表名将会是 movie
    id = db.Column(db.Integer, primary_key=True)  # 主键
    title = db.Column(db.String(60))  # 电影标题
    year = db.Column(db.String(4))  # 电影年份


db.create_all()

# 全局的两个变量移动到这个函数内
name = 'Grey Li'
movies = [
    {'title': 'My Neighbor Totoro', 'year': '1988'},
    {'title': 'Dead Poets Society', 'year': '1989'},
    {'title': 'A Perfect World', 'year': '1993'},
    {'title': 'Leon', 'year': '1994'},
    {'title': 'Mahjong', 'year': '1996'},
    {'title': 'Swallowtail Butterfly', 'year': '1996'},
    {'title': 'King of Comedy', 'year': '1999'},
    {'title': 'Devils on the Doorstep', 'year': '1999'},
    {'title': 'WALL-E', 'year': '2008'},
    {'title': 'The Pork of Music', 'year': '2012'},
]

user = User(name=name)
db.session.add(user)
for m in movies:
    movie = Movie(title=m['title'], year=m['year'])
    db.session.add(movie)

db.session.commit()


@app.route('/user/<name>', methods=['GET'])
def user_page(name):
    return f'User: {escape(name)}'

