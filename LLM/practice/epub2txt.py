# pip install ebooklib
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
def epub_to_text(epub_path):
    book = epub.read_epub(epub_path)
    texts = ''
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.content, 'html.parser')
        texts += soup.get_text()
    return texts

epub_file_path = "./data/jinyong.epub"
output_file_path = "./data/jinyong.txt"

text_content = epub_to_text(epub_file_path)

with open(output_file_path, 'w', encoding='utf-8') as txt_file:
    txt_file.write(text_content)