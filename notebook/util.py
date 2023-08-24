from collections import Counter
from typing import  Callable, Generator, List
from functools import cache


class Vocabulary:
    
    """
    Class for storing a vocabulary of words and their corresponding indices.

    Args:
        special_tokens: A list of special tokens to add to the vocabulary .
        Note: special_tokens will not be processed by preprocessing function.

    Attributes:
        word_to_idx: A dictionary mapping words to their indices.
        idx_to_word: A dictionary mapping indices to their words.
        counter: A counter of the number of times each word appears in the data.
        UNK_TOKEN: The token for unknown words.
        UNK: The index of the UNK_TOKEN.
        PAD_TOKEN: The token for padding.
        PAD: The index of the PAD_TOKEN.
        vocab_size: The size of the vocabulary.

    Methods:
        build_vocab: Build the vocabulary from a dataset of text.
        add_word_to_vocab: Add a word to the vocabulary.
        __len__: Get the length of the vocabulary.
    """
    
    def __init__(self, special_tokens: list[str]):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.counter = Counter()
        
        self.UNK_TOKEN = '<UNK>'
        self.UNK = 1
        self.PAD_TOKEN = '<PAD>'
        self.PAD = 0
        
        self.word_to_idx[self.UNK_TOKEN] = self.UNK
        self.idx_to_word[self.UNK] = self.UNK_TOKEN

        self.word_to_idx[self.PAD_TOKEN] = self.PAD
        self.idx_to_word[self.PAD] = self.PAD_TOKEN

        self.vocab_size = 2
        for idx, token in enumerate(special_tokens, start=2):
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            self.vocab_size += 1

    def build_vocab(self, tokenized_data, max_tokens, min_freq):
        self.counter = Counter()
        for words in tokenized_data:
            self.counter.update(words)

        if max_tokens is not None:
            sorted_tokens = [word for word, _ in self.counter.most_common()]
            for word in sorted_tokens:
                if word not in self.word_to_idx:
                    self.add_word_to_vocab(word)

                if self.vocab_size == max_tokens:
                    break
        else:
            for word, freq in self.counter.items():
                if freq >= min_freq and word not in self.word_to_idx:
                    self.add_word_to_vocab(word)

    def add_word_to_vocab(self, word):
        self.word_to_idx[word] = self.vocab_size
        self.idx_to_word[self.vocab_size] = word
        self.vocab_size += 1

    def __len__(self):
        return self.vocab_size
    

class IntegerVectorizer:
    """
    Class for converting text data to integer vectors.

    Args:
        - tokenizer: A function that takes a string and returns a list of tokens. (default=None)
        - preprocessing_func: A function that takes a token and returns a processed token. (default=None)
        - max_tokens: The maximum number of tokens to keep in the vocabulary. (default=None)
        - min_freq: The minimum frequency of a token to keep in the vocabulary. (default=1)
        - special_tokens: A list of special tokens to add to the vocabulary. (default=None)
        - max_seq_length: The maximum sequence length for each input. (default=None)
        - pad_to_max: Whether to pad sequences to the maximum length. (default=False)

    Attributes:
        vocab: The vocabulary used to convert text to integers.
        tokenized_data: The tokenized data used to build the vocabulary.

    Methods:
        - adapt: Adapt the vectorizer to a dataset of text.
        - __call__: Convert text data to integer vectors.
        - preprocess_sentence: Preprocess a sentence for tokenization.
        - tokenize_data_generator: Tokenize a dataset of text and yield the tokens.
        - transform: Convert a dataset of text to integer vectors.
        - adjust_sequence_length: Adjust the length of a sequence to the maximum length.
        - reverse_transform: Convert integer vectors to text.
        - transform_generator: Convert a dataset of text to integer vectors in a generator.
        - reverse_transform_generator: Convert integer vectors to text in a generator.
    """
    
    def __init__(self, 
                 tokenizer: Callable[[str], list[str]] = None,
                 preprocessing_func: Callable[[str], str] = None,
                 max_tokens=None,
                 min_freq=1,
                 special_tokens: list[str] = None,
                 max_seq_length=None,
                 pad_to_max=False):
        
        self.min_freq = min_freq
        self.max_tokens = max_tokens
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.preprocessing_func = preprocessing_func
        self.reserved_tokens = ['<UNK>', '<PAD>']
        self.special_tokens = [token for token in special_tokens if token not in self.reserved_tokens] if special_tokens else []
        self.pad_to_max = pad_to_max  # Store the argument

        self.vocab = Vocabulary(self.special_tokens)
        self.tokenized_data = []

    def adapt(self, data):
        self.tokenized_data = self.tokenize_data_generator(data)
        self.vocab.build_vocab(self.tokenized_data, self.max_tokens, self.min_freq)
        print('Vocab size:', len(self.vocab))

    def __call__(self, data, reverse=False, return_generator = True):
        if reverse:
            return self.reverse_transform_generator(data) if return_generator else self.reverse_transform(data)
        else:
            return self.transform_generator(data) if return_generator else self.transform(data)

    def preprocess_sentence(self, sentence):
        if self.preprocessing_func:
            words = sentence.split()
            preprocessed_words = [self.preprocessing_func(word) if word not in self.special_tokens else word for word in words]
            return " ".join(preprocessed_words)
        return sentence

    def tokenize_data_generator(self, data):
        for sentence in data:
            sentence = self.preprocess_sentence(sentence)
            yield self.tokenizer(sentence) if self.tokenizer else str(sentence).split()

    def transform(self, data:List[str]):
        
        if not isinstance(data, list):
            raise TypeError("Input data must be a list")
        
        self.tokenized_data = self.tokenize_data_generator(data)
        vectorized_data = []
        for sentence in self.tokenized_data:
            vectorized_sentence = [self.vocab.word_to_idx.get(word, self.vocab.UNK) for word in sentence]
            vectorized_sentence = self.adjust_sequence_length(vectorized_sentence)
            vectorized_data.append(vectorized_sentence)
        return vectorized_data

    def adjust_sequence_length(self, sequence: Generator[int, None, None]) -> list[int]:
        if self.max_seq_length is not None:
            
            if isinstance(sequence, Generator):                
                sequence = list(sequence)
                
            if len(sequence) < self.max_seq_length:
                if self.pad_to_max:
                    sequence += [self.vocab.PAD] * (self.max_seq_length - len(sequence))
            elif len(sequence) > self.max_seq_length:
                sequence = sequence[:self.max_seq_length]
            return sequence

    def reverse_transform(self, vectorized_data: list[list[int]]) -> list[str]:
        original_data = []
        for vector in vectorized_data:
            sentence = [self.vocab.idx_to_word[idx] for idx in vector if idx != self.vocab.PAD]
            original_data.append(" ".join(sentence).strip())
        return original_data


    def transform_generator(self, data: list[str]) -> Generator[list[int], None, None]:
        
        
        if not isinstance(data, list):
            raise TypeError("Input data must be a list")
        
        self.tokenized_data = self.tokenize_data_generator(data)
        for sentence in self.tokenized_data:
            vectorized_sentence = (self.vocab.word_to_idx.get(word, self.vocab.UNK) for word in sentence)
            vectorized_sentence = self.adjust_sequence_length(vectorized_sentence)
            yield list(vectorized_sentence)  # Convert the generator to a list for yielding

    def reverse_transform_generator(self, vectorized_data: list[list[int]]) -> Generator[str, None, None]:
        for vector in vectorized_data:
            sentence = (self.vocab.idx_to_word[idx] for idx in vector if idx != self.vocab.PAD)
            yield " ".join(sentence).strip()