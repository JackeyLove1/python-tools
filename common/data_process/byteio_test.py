import unittest
import os
from io import BytesIO
from hashlib import md5
from byteio import *


class TestFileClasses(unittest.TestCase):
    def setUp(self):
        self.test_text = "This is a test text"
        self.text_id = md5(self.test_text.encode('utf-8')).hexdigest()

    def test_txt_file(self):
        txt_file_path = "test.txt"
        txt_io, result = File.read_file_to_byte(txt_file_path)
        self.assertEquals(result, True)
        txt_file = TxtFile.from_bytes(txt_io)
        self.assertEquals(len(txt_file.docs) > 0, True)

    def test_pdf_file(self):
        pdf_file_path = "test.pdf"
        pdf_io, result = File.read_file_to_byte(pdf_file_path)
        self.assertEquals(result, True)
        pdf_file = PdfFile.from_bytes(pdf_io)
        self.assertEquals(len(pdf_file.docs) > 0, True)

    def test_ppt_file(self):
        ppt_file_path = "test.pptx"
        ppt_io, result = File.read_file_to_byte(ppt_file_path)
        self.assertEquals(result, True)
        ppt_file = PptFile.from_bytes(ppt_io)
        self.assertEquals(len(ppt_file.docs) > 0, True)

    def test_word_file(self):
        docx_file_path = "test.docx"
        docx_io, result = File.read_file_to_byte(docx_file_path)
        self.assertEquals(result, True)
        docx_file = DocxFile.from_bytes(docx_io)
        self.assertEquals(len(docx_file.docs) > 0, True)

    def test_md_file(self):
        md_file_path = "test.md"
        md_io, result = File.read_file_to_byte(md_file_path)
        self.assertEquals(result, True)
        md_file = MdFile.from_bytes(md_io)
        self.assertEquals(len(md_file.docs) > 0, True)
