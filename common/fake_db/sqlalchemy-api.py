import sqlite3
from sqlalchemy import create_engine
# echo=Ture----echo默认为False，表示不打印执行的SQL语句等较详细的执行信息，改为Ture表示让其打印。
engine = create_engine('sqlite:///article.db')
