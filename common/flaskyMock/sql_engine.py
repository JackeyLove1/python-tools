from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import MetaData

engine = create_engine("sqlite+pysqlite:///:memory:", echo=True)
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("select 'hello world'"))
    print(result.all())

with engine.connect() as conn:
    conn.execute(text("CREATE TABLE some_table(x int, y int)"))
    conn.execute(
        text("INSERT INTO some_table (x, y) VALUES  (:x, :y)"),
        [{"x": 1, "y": 1}, {"x": 2, "y": 4}, {"x": 6, "y": 8}, {"x": 9, "y": 10}],
    )

def print_all():
    print("------------- print all -------------")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT x, y FROM some_table"))
        for row in result:
            print(f"x:{row.x} y:{row.y}")

stmt = text("SELECT x, y FROM some_table WHERE y > :y ORDER BY x, y")
with Session(engine) as session:
    result = session.execute(stmt, {"y" : 6})
    for row in result:
        print(f"x:{row.x} y:{row.y}")
with Session(engine) as session:
    result = session.execute(
        text("UPDATE some_table SET y=:y WHERE x=:x"),
        [{"x" : 9, "y" : 11}, {"x" : 13, "y" : 15}],
    )
    session.commit()
print_all()

metadata_obj = MetaData()
print(metadata_obj)

from sqlalchemy import Table, Column, Integer, String
user_table = Table(
    "user_account",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("name", String(30)),
    Column("fullname", String),
)

from sqlalchemy import ForeignKey
address_table = Table(
    "address",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey("user_account.id"), nullable=False),
    Column("email_address", String, nullable=False),
)
metadata_obj.create_all(engine)

from sqlalchemy import insert
stmt = insert(user_table).values(name="spongebob", fullname="Spongebob Squarepants")
print(stmt)
compiled = stmt.compile()
print(compiled.params)

# execute
with engine.connect() as conn:
    result = conn.execute(stmt)

with engine.connect() as conn:
    result = conn.execute(
        insert(user_table),
        [
            {"name": "sandy", "fullname": "Sandy Cheeks"},
            {"name": "patrick", "fullname": "Patrick Star"},
        ],
    )
    # conn.commit()

with Session(engine) as session:
    result = session.execute(stmt)
    print(result)
    session.commit()

from typing import List
from typing import Optional
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase
class Base(DeclarativeBase):
    pass
class User(Base):
    __tablename__ = "user_account"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]
    addresses: Mapped[List["Address"]] = relationship(back_populates="user")
    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id = mapped_column(ForeignKey("user_account.id"))
    user: Mapped[User] = relationship(back_populates="addresses")
    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r})"

# select
