from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import csv
from collections import Counter
import numpy as np
from torch.nn.utils.rnn import pad_sequence


# instance = 1 data entry = text + label
# Txd  (T = sentence len , d =300)
@dataclass
class Instance:
    text: str
    label: str

# vocab
class NLPDataset(Dataset):
    def __init__(self, file_path, vocab=None):
        self.instances = []
        self.vocab = vocab
        self.label_vocab = LabelVocab()
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                text, label = row[:-1][0], row[-1].strip()

                # TEXT = The movie was good --> SPLIT
                text = text.split()  # split text for tokens
                self.instances.append(Instance(text, label))
    
    # total data entries
    def __len__(self):
        return len(self.instances)

    # FROM VOCAB 
    # Text Tensor: tensor([1, 2, 3]), Label Tensor: tensor([0])  
    def __getitem__(self, idx):
        instance = self.instances[idx]
        text_indices = self.vocab.encode(instance.text) if self.vocab else instance.text
        label_index = self.label_vocab.encode(instance.label) if self.label_vocab else instance.label
        return torch.tensor(text_indices), torch.tensor(label_index)

# frequencies = ['the' : 4, 'cat', 3]
# filtering frequencies (if condition) and adding in global VOCAB
class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        self.itos = ['<PAD>', '<UNK>']
        self.stoi = {'<PAD>': 0, '<UNK>': 1}

        for token, freq in frequencies.most_common():
            if freq < min_freq or (max_size != -1 and len(self.itos) >= max_size):
                break
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
    
    # encode = STOI
    # self.stoi['<UNK>'] = DEFAULT - if token not found
    def encode(self, tokens):
        if isinstance(tokens, list):
            return [self.stoi.get(token, self.stoi['<UNK>']) for token in tokens]
        return self.stoi.get(tokens, self.stoi['<UNK>'])

class LabelVocab:
    def __init__(self):
        self.stoi = {'positive': 0, 'negative': 1}
        self.itos = {0: 'positive', 1: 'negative'}
    
    # 0 or 1
    def encode(self, label):
        return self.stoi[label]
    
    # positive or negative
    def decode(self, index):
        return self.itos[index]

# -----------word : occurrence in FULL DATASET
# the cat sat on the mat  = tokens
# the dog barked at the cat
# the:4 , cat:2, sat:1, on:1, mat:1, dog:1, barked:1, at:1 
# tokens=['the', 'cat']  if min_freq=5
def build_vocab(dataset, max_size=-1, min_freq=1):
    frequencies = Counter(token for instance in dataset.instances for token in instance.text)
    return Vocab(frequencies, max_size, min_freq)

train_dataset = NLPDataset('sst_train_raw.csv')
text_vocab = build_vocab(train_dataset, max_size=15000, min_freq=1)
label_vocab = LabelVocab()
train_dataset.vocab = text_vocab

# test 
instance = train_dataset.instances[3]
print(f"Text: {instance.text}")
print(f"Label: {instance.label}")
# test VOCAB
print(f"Numericalized text: {text_vocab.encode(instance.text)}")
print(f"Numericalized label: {label_vocab.encode(instance.label)}")


# SLIDE 31
# glove file = word : its 300D vector = CHECK!!
# 300D = stable model, the rest overfit or underfit 
# word : 1D , 2D , 3D ....

# words -> token -> vocab -> embedding matrix = E (from GLOVE)

# in E 
# if word in GLOVE then put its vector 
# else rand.normal() distribution
def load_glove_embeddings(file_path, vocab):
    embeddings = np.random.normal(0, 1, (len(vocab.itos), 300))
    embeddings[vocab.stoi['<PAD>']] = np.zeros(300)
    if file_path != "no":
        with open(file_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                if word in vocab.stoi:
                    embeddings[vocab.stoi[word]] = vector
    return torch.tensor(embeddings, dtype=torch.float32)

glove_embeddings = load_glove_embeddings("sst_glove_6b_300d.txt", text_vocab)
embedding_layer = torch.nn.Embedding.from_pretrained(glove_embeddings, padding_idx=text_vocab.stoi['<PAD>'])


# variable length sentences
# FOR BATCHING

# [a,b,c] , [d,e] , [f,g,h,i]  
# lengths =[3,2,4]
# pad_sequence() from torch will pad it with (0 or value given in pad_index)
# output = [a,b,c,0] , [d,e, 0,0], [f,g,h,i]
def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.stack(labels)
    return texts_padded, labels, lengths

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
texts, labels, lengths = next(iter(train_loader))
print(f"Texts: {texts}")
print(f"Labels: {labels}")
print(f"Lengths: {lengths}")