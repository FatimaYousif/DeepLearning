import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data import *

class PTLogreg(nn.Module):
  def __init__(self, D, C):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """
    super(PTLogreg, self).__init__()
    self.W = nn.Parameter(torch.randn(D, C))
    self.b = nn.Parameter(torch.randn(1, C))

  def forward(self, X):
    self.Y_ = torch.mm(X, self.W) + self.b
    self.prob = torch.softmax(self.Y_, dim=1)

  def get_loss(self, X, Yoh_, param_lambda=1e-3):
    # Add regularization in a way that you form the loss as a sum 
    # of cross entropy and the L2 norm of the vectorized weight 
    # matrix multiplied with a hyperparameter param_lambda. 
    vectorized_weights = self.W.view(-1)
    L2 = torch.norm(vectorized_weights, p=2)
    self.loss = (torch.sum(-torch.log(self.prob[Yoh_ > 0])) / X.shape[0]) + 0.5 * param_lambda * L2


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=1e-3):
  """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
  """

  optimizer = optim.SGD([model.W, model.b], lr=param_delta)

  # training loop
  for i in range(param_niter):
    # forward pass
    model.forward(X)
    # loss
    model.get_loss(X, Yoh_, param_lambda)
    # gradient reset
    optimizer.zero_grad()
    # backward pass
    model.loss.backward()
    # parameter update
    optimizer.step()

    if i % 1000 == 0:
      print(f'Iteration: {i}, loss: {model.loss}')

def eval(model, X):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  X_tensor = torch.Tensor(X)
  model.forward(X_tensor)
  return torch.Tensor.numpy(model.prob.detach())

def decfun(model):
    def classify(X):
      return np.argmax(eval(model, X), axis=1)
    return classify

if __name__ == "__main__":
    # initialize random number generator
    np.random.seed(100)

    # define input data X and labels Yoh_
    X, Y_ = sample_gauss_2d(3, 100)
    # X, Y_ = sample_gmm_2d(4, 2, 40)
    Yoh_ = class_to_onehot(Y_)
    # define the model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])


    # learn the parameters (X and Yoh_ have to be of type torch.Tensor):
    # hyper-params : itr, delta (lr), lambda (regularization)

    # GOOD hyperparams
    # train(ptlr, torch.Tensor(X), torch.Tensor(Yoh_), 60000, 0.01, 1e-3)

    # ---unsatisfactory results
    # train(ptlr, torch.Tensor(X), torch.Tensor(Yoh_), 10000, 0.01, 10.0)
    # train(ptlr, torch.Tensor(X), torch.Tensor(Yoh_), 10000, 1.0, 0.001)  # NOT REALLY
    
    # get probabilites on training data
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)

    # print out the performance metric (precision and recall per class)
    accuracy, recall, precision = eval_perf_multi(Y, Y_)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

    # Decicion surface
    decfun = decfun(ptlr)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0)
    graph_data(X, Y_, Y)
    plt.show()