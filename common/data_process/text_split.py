from typing import Dict, List, Optional
from langchain.docstore.document import Document
from abc import ABC, abstractmethod
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import LatexTextSplitter


class BaseSpliter(ABC):
    def __init__(self, content: str):
        self.content = content

    @abstractmethod
    def split_content(self) -> List[str]:
        pass

    @abstractmethod
    def create_document(self) -> List[Document]:
        pass

    # 字符分割


# 固定大小的分块
class CharacterContentSplitter(BaseSpliter):
    def __init__(self, content: str, chunk_size=256, chunk_overlap=20):
        super().__init__(content)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap)

    def split_content(self) -> List[str]:
        return self.splitter.split_text(self.content)

    def create_document(self) -> List[Document]:
        pass

    # NLP语意分割


class NLTKContentSplitter(BaseSpliter):
    def __init__(self, content: str):
        super().__init__(content)
        self.splitter = NLTKTextSplitter()

    def split_content(self) -> List[str]:
        return self.splitter.split_text(self.content)

    def create_document(self) -> List[Document]:
        pass


class SpaCyContentSplitter(BaseSpliter):
    def __init__(self, content: str):
        super().__init__(content)
        self.splitter = SpacyTextSplitter()

    def split_content(self) -> List[str]:
        return self.splitter.split_text(self.content)

    def create_document(self) -> List[Document]:
        pass

    # 递归分块


# 递归分块使用一组分隔符以分层和迭代方式将输入文本划分为较小的块。如果拆分文本的初始尝试未生成所需大小或结构的块，
# 则该方法会使用不同的分隔符或条件递归调用生成的块，直到达到所需的块大小或结构。这意味着，虽然块的大小不会完全相同，
# 但它们仍然追求具有相似的大小。
class RecursiveCharacterContentSplitter(BaseSpliter):
    def __init__(self, content: str, chunk_size=256, chunk_overlap=20):
        super().__init__(content)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap)

    def split_content(self) -> List[str]:
        return self.splitter.split_text(self.content)

    def create_document(self) -> List[Document]:
        pass

    # 特殊文本分块


class MarkdownContentSplitter(BaseSpliter):
    def __init__(self, content: str, chunk_size=100, chunk_overlap=0):
        super().__init__(content)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap)

    def split_content(self) -> List[str]:
        return self.splitter.split_text(self.content)

    def create_document(self) -> List[Document]:
        pass


class LatexTContentSplitter(BaseSpliter):
    def __init__(self, content: str, chunk_size=100, chunk_overlap=0):
        super().__init__(content)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = LatexTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap)

    def split_content(self) -> List[str]:
        return self.splitter.split_text(self.content)

    def create_document(self) -> List[Document]:
        pass
