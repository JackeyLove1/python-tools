from abc import abstractmethod, ABC
from typing import List, Iterator, Generator, Union
from io import BytesIO
import PyPDF2
from PyPDF2 import PdfReader


class Document:
    def __init__(self, content: str, meta: dict):
        self.content = content
        self.meta = meta

class BaseLoader(ABC):
    @abstractmethod
    def loader(self) -> List[Document]:
        """Load data into document objects."""
        pass

    def load_and_split(self):
        pass

    def lazy_loader(self) -> Iterator[Document]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement lazy_load()")


class PyPDFLoader(BaseLoader):
    def __init__(self, file: str):
        self.file = file

    def lazy_loader(self) -> Generator[Document]:
        reader = PdfReader(self.file)

        for page_number in range(reader.getNumPages()):
            page = reader.getPage(page_number)
            text = page.extract_text()
            paragraphs = text.split('\n\n')  # This may vary depending on how paragraphs are separated in your PDF.

            for paragraph_rank, paragraph in enumerate(paragraphs, start=1):
                metadata = {
                    'page_number': page_number + 1,  # Adjusting page_number to start from 1 instead of 0
                    'paragraph_rank': paragraph_rank
                }
                yield Document(paragraph, metadata)

    def loader(self) -> List[Document]:
        raise NotImplementedError()
