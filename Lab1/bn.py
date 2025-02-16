
import torch
import torch.nn as nn
import torchvision
from pt_deep import *
from mnist_shootout import *

# https://medium.com/@krishna_84429/understanding-batch-normalization-1eaca8f2f63e

class PTDeepWithBatchNorm(PTDeep):
    def __init__(self, config, activation=torch.relu, epsilon=1e-5):
        super(PTDeepWithBatchNorm, self).__init__(config, activation)
        self.epsilon = epsilon
        self.hidden_layers = self.layers - 1

        # Add batch normalization parameters only for hidden layers
        self.gamma = nn.ParameterList([nn.Parameter(torch.ones(1, config[i])) for i in range(1, self.layers)])
        self.beta = nn.ParameterList([nn.Parameter(torch.zeros(1, config[i])) for i in range(1, self.layers)])
        self.running_mean = [torch.zeros(1, config[i]) for i in range(1, self.layers)]
        self.running_var = [torch.ones(1, config[i]) for i in range(1, self.layers)]

    def batch_norm(self, x, gamma, beta, layer_idx):
        if self.training:
            # Batch normalization during training
            mean = torch.mean(x, dim=0, keepdim=True)
            var = torch.var(x, dim=0, unbiased=False, keepdim=True)
            x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)

            # Update running mean and variance only for hidden layers
            self.running_mean[layer_idx] = (1 - 0.9) * self.running_mean[layer_idx] + 0.9 * mean
            self.running_var[layer_idx] = (1 - 0.9) * self.running_var[layer_idx] + 0.9 * var
        else:
            # Batch normalization during inference using running mean and variance
            x_normalized = (x - self.running_mean[layer_idx]) / torch.sqrt(self.running_var[layer_idx] + self.epsilon)

        return gamma * x_normalized + beta

    def forward(self, X):
        self.Y_ = X
        for i in range(self.layers):
            # Affine transformation
            affine_output = torch.mm(self.Y_, self.weights[i]) + self.biases[i]

            # Batch normalization only after hidden layers
            if self.hidden_layers > 0 and i != self.layers - 1:
                affine_output = self.batch_norm(affine_output, self.gamma[i], self.beta[i], i)

            # Activation function
            if i != self.layers - 1:
                self.Y_ = self.activation(affine_output)
            else:
                max_values, indices = torch.max(affine_output, dim=1)
                max_values = max_values.view(-1, 1)
                self.Y_ = affine_output - max_values
                self.Y_ = self.Y_.double()
                self.prob = torch.softmax(self.Y_, dim=1)

    def get_loss(self, X, Yoh_, param_lambda=1e-3):
        vectorized_weights = torch.cat([self.weights[i].view(-1) for i in range(self.layers)])
        L2 = torch.norm(vectorized_weights, p=2)
        self.loss = torch.mean(-torch.log(self.prob[Yoh_ > 0])) + (param_lambda * L2)


if __name__ == '__main__':
    dataset_root = 'FCNNs/mnist'  
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    N=x_train.shape[0]
    D=x_train.shape[1]*x_train.shape[2]
    C=y_train.max().add_(1).item()

    y_train_oh = class_to_onehot(y_train)


    ptlr = PTDeepWithBatchNorm([D, 100, C])

    stored_loss = train_mb(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 1000, 0.1, 0.1, 100, optimizer=optim.SGD)
    plt.plot(stored_loss)
    plt.show()

    # get probabilites on training data
    print('Training set Metrics')
    probs = eval(ptlr, x_train.view(N, D))
    Y = np.argmax(probs, axis=1)
    # print out the performance metric (precision and recall per class)
    accuracy, recall, precision = eval_perf_multi(Y, y_train)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))
