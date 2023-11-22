# mock a simple tokenizer
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple, Any, List


class Tokenizer(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def tokenize(self, text: str):
        """Tokenize the text, converting it into a list of tokens (chars in our case). """
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts a sequence of tokens into ids using the vocab."""
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts a sequence of ids in tokens using the vocab."""
        pass


class CalTokenizer(Tokenizer):
    def __init__(self):
        self.stoi = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '9': 9,
            '+': 10,
            '-': 11,
            '*': 12,
            '/': 13,
            '=': 14,
            ',': 15,
            ':': 16,
            '_': 17,
            '(': 18,
            ')': 19,
            ' ': 20,
            'P': 21,  # PAD
            'S': 22,  # SEP
            'E': 23,  # EOS
            'U': 24,  # UNKNOWN
            '^': 25,
            '!': 26
        }

        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    @property
    def vocab_size(self):
        return self.vocab_size

    @property
    def stoi(self):
        return self.stoi

    @property
    def itos(self):
        return self.itos

    @property
    def pad_token_id(self):
        return self.stoi['P']

