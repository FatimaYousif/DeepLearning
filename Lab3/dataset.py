import numpy as np
import torch
from torch.utils.data import DataLoader
from instance import Instance
from vocab import *
from torch.utils.data import Dataset
from typing import List

import csv
from torch.utils.data import Dataset
from collections import Counter

class NLPDataset(Dataset):
    def __init__(self, file_path, text_vocab=None, label_vocab=None):
        self.instances = []
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab or LabelVocab()
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                text, label = row[:-1][0], row[-1].strip()
                text = text.split()  # text into tokens
                self.instances.append(Instance(text, label))
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        text_indices = torch.tensor(self.text_vocab.encode(instance.text), dtype=torch.long)
        label_index = torch.tensor(self.label_vocab.encode(instance.label), dtype=torch.float)
        return text_indices, label_index

    @staticmethod
    def build_frequencies(texts: List[List[str]]) -> Counter:
        return Counter(token for text in texts for token in text)

    @staticmethod
    def build_vocab(dataset, max_size=-1, min_freq=1):
        frequencies = NLPDataset.build_frequencies([instance.text for instance in dataset.instances])
        return Vocab(frequencies, max_size, min_freq)

def load_glove_embeddings(file_path, vocab, embedding_dim=300):
    embeddings = np.random.normal(0, 1, (len(vocab.itos), embedding_dim)).astype(np.float32)
    embeddings[vocab.stoi['<PAD>']] = np.zeros(embedding_dim)
    if file_path != "normal":
        # print("hereee")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                if word in vocab.stoi:
                    embeddings[vocab.stoi[word]] = vector
    return torch.tensor(embeddings, dtype=torch.float32)

def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.stack(labels)
    # labels = torch.stack(labels).view(-1, 1)  # for task 3
    return texts_padded, labels, lengths


train_path = 'sst_train_raw.csv'
valid_path = 'sst_valid_raw.csv'
test_path = 'sst_test_raw.csv'
glove_path = 'sst_glove_6b_300d.txt'

train_dataset = NLPDataset(train_path)

text_vocab = NLPDataset.build_vocab(train_dataset, max_size=-1, min_freq=1)

# vocab size 12k
# text_vocab = NLPDataset.build_vocab(train_dataset, max_size=12000, min_freq=1)

label_vocab = LabelVocab()
train_dataset.text_vocab = text_vocab

valid_dataset = NLPDataset(valid_path, text_vocab, label_vocab)
test_dataset = NLPDataset(test_path, text_vocab, label_vocab)

# original
embedding_layer = torch.nn.Embedding.from_pretrained(load_glove_embeddings(glove_path, text_vocab), padding_idx=text_vocab.stoi['<PAD>'], freeze=True)

# freeze false + if no embeddings
# embedding_layer = torch.nn.Embedding.from_pretrained(load_glove_embeddings("normal", text_vocab), padding_idx=text_vocab.stoi['<PAD>'], freeze=False)
