import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data import *


class PTDeep(nn.Module):
    # CONFIG ---given
    def __init__(self, config, activation=torch.relu):
        super(PTDeep, self).__init__()
        self.layers = len(config) - 1

        # lab req....
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(config[i], config[i+1])) for i in range(self.layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(1, config[i+1])) for i in range(self.layers)])
        self.activation = activation

        # NEEDED FOR MNIST NEXT EXERCISE -------
        self.most_loss_images = []
        self.loss_images_num = 10
        
    def forward(self, X):
        self.Y_ = X
        for i in range(self.layers):
            self.Y_ = torch.mm(self.Y_, self.weights[i]) + self.biases[i]
            if i != self.layers - 1:
                self.Y_ = self.activation(self.Y_)
            else:
                max_values, indices = torch.max(self.Y_, dim=1)
                max_values = max_values.view(-1, 1)
                self.Y_ = self.Y_ - max_values
                # Give more precision to the output so it does not go to zero
                self.Y_ = self.Y_.double()
                self.prob = torch.softmax(self.Y_, dim=1)

        # -----for WITHOUT TRAINING PART in MNIST SHOOTOUT lab
        # return self.prob

    def get_loss(self, X, Yoh_, param_lambda=1e-3):
        # Add regularization in a way that you form the loss as a sum 
        # of cross entropy and the L2 norm of the vectorized weight 
        # matrix multiplied with a hyperparameter param_lambda. 
        # ----
        vectorized_weights = torch.cat([self.weights[i].view(-1) for i in range(self.layers)])
        L2 = torch.norm(vectorized_weights, p=2)
        negative_log_likelihood = -torch.log(self.prob[Yoh_ > 0])
        self.loss = torch.mean(negative_log_likelihood) + (param_lambda * L2)

        # Get the images which contribute the most to the loss
        top_values, top_indices = torch.topk(torch.abs(negative_log_likelihood), k=self.loss_images_num)
        combined_list = [(loss, index) for loss, index in zip(top_values.detach().numpy(), top_indices.detach().numpy())]
        self.most_loss_images.extend(combined_list)

        # Remove duplicates based on index
        unique_loss_images = {}
        for loss, index in self.most_loss_images:
            if index not in unique_loss_images:
                unique_loss_images[index] = loss

        self.most_loss_images = [(loss, index) for index, loss in unique_loss_images.items()]
        self.most_loss_images.sort(key=lambda x: x[0], reverse=True)
        if len(self.most_loss_images) > self.loss_images_num:
            self.most_loss_images = self.most_loss_images[:self.loss_images_num]
        

        #---------- FOR WITHOUT TRAIN ------------------------------- 
        # vectorized_weights = torch.cat([self.weights[i].view(-1) for i in range(self.layers)])
        # L2 = torch.norm(vectorized_weights, p=2)

        # # Softmax to get probabilities from logits
        # self.prob = torch.softmax(X, dim=1)

        # # Cross-entropy loss: using the one-hot encoded labels and probabilities
        # # The cross-entropy loss is the negative log of the predicted probability of the true class
        # log_probs = torch.log(self.prob + 1e-13)  # Small epsilon to avoid log(0)
        # cross_entropy_loss = -torch.sum(Yoh_ * log_probs, dim=1)  # Sum across the classes

        # # Mean of the cross-entropy loss
        # negative_log_likelihood = torch.mean(cross_entropy_loss)

        # # Total loss = cross-entropy loss + regularization term
        # self.loss = negative_log_likelihood + param_lambda * L2

        # return self.loss

        # ----------------------------------------------

def train(model, X, Yoh_, param_niter, param_delta, param_lambda=1e-3, optimizer=optim.SGD):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    optimizer = optimizer(model.parameters(), lr=param_delta)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)
    stored_loss = []

    # training loop
    for i in range(param_niter):
        # forward pass
        model.forward(X)
        # loss
        model.get_loss(X, Yoh_, param_lambda=param_lambda)
        stored_loss.append(model.loss.detach().numpy())
        # gradient reset
        optimizer.zero_grad()
        # backward pass
        model.loss.backward()
        # parameter update
        optimizer.step()
        if i % 1000 == 0 and i!=0:
            print(f'Iteration: {i}, loss: {model.loss}')

    # MNIST task ----------
    if optimizer.__class__.__name__ == 'Adam':
        scheduler.step()
        # print("scheduler used .......")
    
    return stored_loss

def eval(model, X):
    """
    Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
    """
    X_tensor = torch.Tensor(X)
    model.forward(X_tensor)
    return torch.Tensor.numpy(model.prob.detach())

def decfun(model):
    def classify(X):
      return np.argmax(eval(model, X), axis=1)
    return classify


def count_params(model):
    total_params = 0
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Size: {param.size()}")
        total_params += param.numel()
    return total_params

if __name__== "__main__":

    # initialize random number generator
    np.random.seed(100)

    # define input data X and labels Yoh_
    # X, Y_ = sample_gauss_2d(3, 100)

    X, Y_ = sample_gmm_2d(6, 2, 10)
    # X, Y_ = sample_gmm_2d(4, 2, 40)
    
    
    Yoh_ = class_to_onehot(Y_)
    
    # 2D data, 3 classes
    # ...2 neurons mapped to 3 neurons 
    # xw+b = 2 x 3 +3 = 9 (#params) 
    # ptlr = PTDeep([2, 3], torch.relu)

    # TODO --------------------------
    # config 1
    ptlr = PTDeep([2, 2], torch.relu)

    # config 2
    # ptlr = PTDeep([2, 10, 2], torch.relu)

    # config 3
    # ptlr = PTDeep([2, 10, 10, 2], torch.relu)

    # SIGMOID................. grad(relu) doesnt saturate
    # ptlr = PTDeep([2,10,2], torch.sigmoid)
    # --------------------------

    # learn the parameters (X and Yoh_ have to be of type torch.Tensor):
    train(ptlr, torch.Tensor(X), torch.Tensor(Yoh_), 10000, param_delta=0.1, param_lambda=1e-4)

    # get probabilites on training data
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)
    
    
    # print out the performance metric (precision and recall per class)
    accuracy, recall, precision = eval_perf_multi(Y, Y_)
    average_precision = eval_AP(Y)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))
    print("Average precision: {}".format(average_precision))
    print("Total number of parameters: ",count_params(ptlr))

    # Decicion surface
    decfun = decfun(ptlr)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0)
    graph_data(X, Y_, Y)
    plt.show()