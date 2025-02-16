import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from dataset import *
from task2_baseline import *

class VanillaRNNModel(nn.Module):
    def __init__(self, embedding_layer, hidden_size=150, num_layers=2):
        super(VanillaRNNModel, self).__init__()
        self.embedding_layer = embedding_layer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Vanilla RNN
        self.rnn1 = nn.RNN(input_size=embedding_layer.embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=False)  # time-first format
        
        self.rnn2 = nn.RNN(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=False)
        
        # Fully connected 
        self.fc1 = nn.Linear(hidden_size, 150)
        self.fc2 = nn.Linear(150, 1)

    def forward(self, x):

        embedded = self.embedding_layer(x)
        embedded = embedded.transpose(0, 1)

        # ENCODER 
        # Pass through VANILLA RNN layers
        output, hidden1 = self.rnn1(embedded)        
        output, hidden2 = self.rnn2(output)
        
        # get last hidden state
        hidden = hidden2[-1]
        
        # DECODER 
        # pass that hidden -> FC 
        h1 = self.fc1(hidden)
        h2 = F.relu(h1)
        h3 = self.fc2(h2)
        return h3
    

def train_vanilla_rnn(seed, epochs=5, batch_size=10, hidden_size=150, num_layers=2):
    print(f"Seed {seed}:")
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, collate_fn=pad_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, 
                                shuffle=False, collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, collate_fn=pad_collate_fn)

    model = VanillaRNNModel(embedding_layer, hidden_size=hidden_size, num_layers=num_layers)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        train(model, train_dataloader, optimizer, criterion, clip=0.25)
        avg_loss, accuracy, f1, conf_matrix = evaluate(model, valid_dataloader, criterion)
        print("Epoch {}: valid accuracy = {}".format(epoch, accuracy))

    avg_loss, accuracy, f1, conf_matrix = evaluate(model, test_dataloader, criterion)
    print("Avg_loss: ", avg_loss)
    print("f1:", f1) 
    print("Confusion matrix:\n", conf_matrix)
    print("Test accuracy = {}".format(accuracy))

if __name__ == '__main__':
    # 5x diff
    for seed in range(1, 6):
        print(f"\nTraining run {seed}/5")
        print("-" * 50)
        train_vanilla_rnn(seed)
    # train_vanilla_rnn(7052020)
    