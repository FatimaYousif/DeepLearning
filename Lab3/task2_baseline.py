import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataset import *
from sklearn.metrics import f1_score, confusion_matrix

# SEQUENCE CLASSIFICATION - only last hidden layer
class BaselineModel(nn.Module):
    def __init__(self, embedding_layer):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)
        self.embedding_layer = embedding_layer

    # archi given
    def forward(self, x):
        x = self.embedding_layer(x)
        h1 = x.mean(dim = 1)  # mean pooling = req
        h2 = self.fc1(h1)
        h3 = F.relu(h2)
        h4 = self.fc2(h3)
        h5 = F.relu(h4)
        h6 = self.fc3(h5)
        return h6


def train(model, data, optimizer, criterion, clip=-1):
    model.train()

    # batch = bcz pad_collate_fn(batch)
    for batch in data:

        # from pad_collate = padded_text, label, lengths
        x, y, lengths = batch
        model.zero_grad()
        logits = model(x)
        logits = model.forward(x)
        logits = logits.squeeze()
        loss = criterion(logits, y)
        loss.backward()

        # why ?? = to rmv exploding gradients
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def evaluate(model, data, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data:
            x, y, lengths = batch
            logits = model(x)
            # # logits = logits.squeeze()     #task 2
            # y = y.view(-1, 1)               #task3
            logits = logits.squeeze(-1)     #for all
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(logits))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    avg_loss = total_loss / len(data)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, conf_matrix

def main_base(seed, epochs, batch_size):

    # for TESTING MODEL 5x
    print("Seed {}:".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True, collate_fn=pad_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size[1], shuffle=False, collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size[2], shuffle=False, collate_fn=pad_collate_fn)

    model = BaselineModel(embedding_layer)

    # suggested - binary CE
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        train(model, train_dataloader, optimizer, criterion)
        avg_loss, accuracy, f1, conf_matrix = evaluate(model, valid_dataloader, criterion)
        print("Epoch {}: valid accuracy = {}".format(epoch, accuracy))

    avg_loss, accuracy, f1, conf_matrix = evaluate(model, test_dataloader, criterion)
    print("Avg_loss: ", avg_loss)
    print("f1:", f1) 
    print("Confusion matrix:\n", conf_matrix)
    print("Test accuracy = {}".format(accuracy))

if __name__ == '__main__':

    # req = model stability= 5x on different seeds
    # for i in range(1, 6):
    #     main_base(i, 5, [32, 32, 32])

    # main_base(7052020, 5, [10,10,10])   # lab match
    
    # without pretrained embeddings
    main_base(7052020, 5, [32, 32, 32])   
    