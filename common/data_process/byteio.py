# Write the modified data back to the file
'''
with open(file_path, 'wb') as f:
    f.write(modified_data)
with open(file_path, 'rb') as f:
    # Read the data from the file
    data = f.read()
    # Create a BytesIO object from the data
    byte_stream = BytesIO(data)
    modified_data = byte_stream.getvalue()
    print(modified_data)
提取某一个文件所有的文本，然后交给langchain切割文本的函数进行流水线处理
'''
import io
from io import BytesIO
import os

import docx
import langchain
import openai
from typing import List, Any, Optional
import re
import docx2txt
from langchain.docstore.document import Document
import fitz
from hashlib import md5
from abc import abstractmethod, ABC
from copy import deepcopy
import docx2txt


class File(ABC):
    def __init__(self,
                 id: str,
                 name: Optional[str] = None,
                 meta: Optional[dict[str, Any]] = None,
                 docs: Optional[str] = None):
        self.id = id
        self.name = name or ""
        self.meta = meta or {}
        self.docs = docs or ""  # all raw text

    @classmethod
    @abstractmethod
    def from_bytes(cls, files: BytesIO) -> "File":
        pass

    @staticmethod
    def read_file_to_byte(file_path: str) -> (BytesIO, bool):
        byte_io = None
        try:
            with open(file_path, 'rb') as file:
                byte_io = io.BytesIO(file.read())
            byte_io.seek(0)
            return byte_io, True
        except Exception as e:
            return byte_io, False

    def __repr__(self):
        return f"name:{self.name}, id:{self.id}, docs:{str(self.docs)}"


def strip_consecutive_newlines(text: str) -> str:
    """Strips consecutive newlines from a string
    possibly with whitespace in between
    """
    return re.sub(r"\s*\n\s*", "\n", text)


# package list: docx(complex), docx2txt(simple)
class DocxFile(File):
    @classmethod
    def from_bytes(cls, files: BytesIO) -> "DocxFile":
        text = docx2txt.process(files)
        docs = strip_consecutive_newlines(text)
        return cls(id=md5(files.read()).hexdigest(), docs=docs)


class TxtFile(File):
    @classmethod
    def from_bytes(cls, files: BytesIO) -> "TxtFile":
        files.seek(0)
        text = files.read().decode("utf-8")
        text = strip_consecutive_newlines(text)
        docs = text.strip()  # TODO: consider "\n"
        return cls(id=md5(files.read()).hexdigest(), docs=docs)


docx_file_path = "test.docx"
docx_file, _ = File.read_file_to_byte(docx_file_path)
print(DocxFile.from_bytes(docx_file))

txt_file_path = "test.txt"
file, _ = File.read_file_to_byte(txt_file_path)
print(TxtFile.from_bytes(file))
