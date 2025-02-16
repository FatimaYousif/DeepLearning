import torch
import time
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from util_plot import *
from load_ciphar10 import load_cifar10
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

SAVE_DIR = Path(__file__).parent / 'out_c2'
np.random.seed(int(time.time() * 1e6) % 2**31)

config = {
    'max_epochs': 50,
    'batch_size': 64,
    'lr': 0.01,
    'lr_gamma': 0.95,
    'weight_decay': 1e-4,
}

plot_data = {
    'train_loss': [],
    'valid_loss': [],
    'train_acc': [],
    'valid_acc': [],
    'lr': []
}

# input image =32 x 32
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(7 * 7 * 32, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 10, bias=True)
        self.fc_logits = nn.Linear(128, 10, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_logits(x)
        return x

def train(train_set, valid_set, model):
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ExponentialLR(optimizer, gamma=config['lr_gamma'])
    
    for epoch in range(1, config['max_epochs'] + 1):                    
        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #  loss = multiclass_hinge_loss(outputs, labels, delta=1.0)     #bonus
            loss.backward()
            optimizer.step()
            
            if i % 200 == 0:
                print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, i * config['batch_size'], len(train_loader.dataset), loss.item()))
        
        scheduler.step()

        # evaluate
        train_loss, train_acc = evaluate("Train:", model, train_loader)
        val_loss, val_acc = evaluate("Validation:", model, valid_loader)

        # draw filters = FIRST and LAST
        if epoch == 1 or epoch == config['max_epochs']:
            draw_conv_filters(epoch, 45000, model.conv1.weight, SAVE_DIR)
        
        # store for loss plot
        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]
        plot_data['lr'] += [scheduler.get_last_lr()]

def evaluate(name, model, loader):
    model.eval()
    
    # for confusion matrix
    all_predictions = []
    all_labels = []

    # for biggest loss images
    sample_losses = []
    highest_probs = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sample_losses.extend(loss.cpu().numpy())
            
            probs = F.softmax(outputs, dim=1)
            highest_probs.extend(probs.cpu().numpy())
            _, predicted = torch.max(probs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    average_loss = np.mean(sample_losses)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    print(name)
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Average Loss: {average_loss}")
    print(f"Accuracy: {accuracy}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}\n")

    # pass it to plot_images()
    # name can help reuse this method tor train & valid too
    if name == "Test:":
        top_20_loss_indexes = np.argsort(sample_losses)[-20:]
        highest_probs = np.array(highest_probs)
        highest_probs = highest_probs[top_20_loss_indexes]
        return top_20_loss_indexes, highest_probs
    return average_loss, accuracy

# BONUS
def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta=1.0):
    batch_size = logits.size(0)
    correct_logits = logits[torch.arange(batch_size), target].view(-1, 1)
    margins = logits - correct_logits + delta
    margins[torch.arange(batch_size), target] = 0
    loss = torch.max(torch.zeros_like(margins), margins).sum() / batch_size     # YT vid
    print(loss)
    return loss

if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset, data_mean, data_std = load_cifar10()
    model = ConvNet()
    train(train_dataset, valid_dataset, model)
    plot_training_progress(SAVE_DIR, plot_data)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    top_losses, probs = evaluate("Test:", model, test_loader)
    plot_images(SAVE_DIR, test_dataset, top_losses, probs, data_mean, data_std)