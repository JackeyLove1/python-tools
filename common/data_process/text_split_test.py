from text_split import *
import unittest


class TestTextSpliter(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk_size = 1
        self.chunk_overlap = 0
        self.content = "Hello\n\n How\n\n are\n\n you\n\n?"
        self.length = 5

    def test_CharacterContentSplitter(self):
        spliter = CharacterContentSplitter(self.chunk_size, self.chunk_overlap)
        items = spliter.split_content(self.content)
        self.assertEquals(len(items), self.length)

    def test_RecursiveCharacterContentSplitter(self):
        spliter = RecursiveCharacterContentSplitter()
        items = spliter.split_content(self.content)

