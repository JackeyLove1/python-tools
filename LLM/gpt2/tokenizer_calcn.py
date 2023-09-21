class CalcNTokenizer:
    """Calc N Tokenizer."""

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
        self.itos = {
            value: key for key, value in self.stoi.items()
        }
        self.vocab_size = 27
        assert self.vocab_size == len(self.stoi.keys())

    @property
    def pad_token_id(self):
        return self.stoi['P']

    @property
    def eos_token_id(self):
        return self.stoi['E']

    @property
    def sep_token_id(self):
        return self.stoi['S']

    def tokenize(self, text):
        """Tokenize the text, converting it into a list of tokens (chars in our case). """
        output = [c for c in text]
        return output

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        unknown_id = self.stoi['U']
        return [self.stoi.get(tok, unknown_id) for tok in tokens]

    def encode(self, text):
        """Convert the string text to a list of tokenized ids.
        """
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, ids):
        """Converts a sequence of ids to a string"""
        if not isinstance(ids, list):
            ids = [ids]  # at least a length one list
        output = [self.itos.get(item, 'U') for item in ids]
        output = ''.join(output)
        return output
