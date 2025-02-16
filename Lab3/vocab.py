import torch

class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=1):
        self.max_size = max_size
        self.min_freq = min_freq
        
        self.itos, self.stoi = {}, {}
        self.special_tokens = ["<PAD>", "<UNK>"]
        for i, tok in enumerate(self.special_tokens):
            self.itos[i] = tok
            self.stoi[tok] = i
        
        self._build_vocab(frequencies)

    def _build_vocab(self, frequencies):
        sorted_words = sorted(frequencies.items(), key=lambda item: -item[1])
        sorted_words = [word for word, freq in sorted_words if freq >= self.min_freq]
        
        if self.max_size != -1:
            sorted_words = sorted_words[:self.max_size - len(self.special_tokens)]
        
        start_index = len(self.itos)
        for idx, word in enumerate(sorted_words, start_index):
            self.itos[idx] = word
            self.stoi[word] = idx

    def encode(self, tokens):
        if isinstance(tokens, list):  # more 
            return [self.stoi.get(token, self.stoi['<UNK>']) for token in tokens]
        return self.stoi.get(tokens, self.stoi['<UNK>'])  # one

    def decode(self, indices):
        if isinstance(indices, list):  # more
            return [self.itos.get(index, '<UNK>') for index in indices]
        return self.itos.get(indices, '<UNK>')  # one

    def __len__(self):
        return len(self.itos)
    
class LabelVocab:
    def __init__(self):
        self.stoi = {'positive': 0, 'negative': 1}
        self.itos = {0: 'positive', 1: 'negative'}
    
    def encode(self, label):
        return self.stoi[label]
    
    def decode(self, index):
        return self.itos[index]