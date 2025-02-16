import numpy as np
from data import *

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps, axis=1, keepdims=True)

# -----TODO function
# X = (N,D) = MNIST:
# N = #imgs 
# D = features= 28x28 
def fcann2_train(X, Y_, param_niter = 100000, param_delta = 0.05, param_lambda = 1e-3, param_hidden_layer_size = 5):
    C = int(max(Y_) + 1)  #if 3 for (0,1,2)
    N = X.shape[0]     #BGD
    W1 = np.random.randn(X.shape[1], param_hidden_layer_size) # D x N (Neurons)
    b1 = np.zeros((1, param_hidden_layer_size)) # 1 x N
    W2 = np.random.randn(param_hidden_layer_size, C) # N x C
    b2 = np.zeros((1, C)) # 1 x C
    
    for i in range(param_niter):
        # forward pass
        scores1 = np.dot(X, W1) + b1
        hidden_layer = np.maximum(0, scores1)
        scores2 = np.dot(hidden_layer, W2) + b2
        probs = softmax(scores2)

        one_hot = class_to_onehot(Y_)

        # NLL loss
        data_loss = -np.sum(one_hot * np.log(probs)) / N
        reg_loss = 0.5 * param_lambda * np.sum(W1 * W1) + 0.5 * param_lambda * np.sum(W2 * W2)  #L2 regularization
        loss = data_loss + reg_loss

        # every 1000th iteration results
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # backpropagation
        dL_dscores2 = probs - one_hot
        dL_dW2 = np.dot(hidden_layer.T, dL_dscores2) / N + param_lambda * W2  #param_lambda = for small w (NO overfit)
        dL_db2 = np.sum(dL_dscores2, axis=0, keepdims=True) / N
        dL_dhidden_layer = np.dot(dL_dscores2, W2.T)
        dL_dscores1 = dL_dhidden_layer
        dL_dscores1[scores1 <= 0] = 0
        dL_dW1 = np.dot(X.T, dL_dscores1) / N + param_lambda * W1
        dL_db1 = np.sum(dL_dscores1, axis=0, keepdims=True) / N

        # update parameters
        # param_delta = learning rate 
        # The subtraction ensures that we move in the direction of decreasing the loss
        # ___ optimization.step()
        W1 -= param_delta * dL_dW1
        b1 -= param_delta * dL_db1
        W2 -= param_delta * dL_dW2
        b2 -= param_delta * dL_db2

    return W1, b1, W2, b2

# -------TODO function
def fcann2_classify(X, W1, b1, W2, b2):
    scores1 = np.dot(X, W1) + b1
    hidden_layer = np.maximum(0, scores1)
    scores2 = np.dot(hidden_layer, W2) + b2
    probs = softmax(scores2)
    return probs


# LAB 1
def decfun(W1, b1, W2, b2):
    def classify(X):
      return np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # ---- sample_gmm = components, classes, samples = CCS
    # ---- sample_gauss = classes, samples = CS
    
    # create the inputs and labels
    X, Y_ = sample_gmm_2d(6, 2, 10)
    # Shape of X: (N, D) N = number of samples = ncomponents x nsamples,
    # D = dimensionality of each sample = 2 for 2d data
    # Shape of Y_: (N,)

    # ------ GIVEN FROM LAB HANDOUT
    # training
    W1, b1, W2, b2 = fcann2_train(X, Y_, param_niter=100000, param_delta=0.05, param_lambda=1e-3, param_hidden_layer_size=5)

    # model  evaluation on the training dataset
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)
    accuracy, recall, precision = eval_perf_multi(Y, Y_)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

    # graph the decision surface
    decfun = decfun(W1, b1, W2, b2)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0)
    graph_data(X, Y_, Y)
    plt.show()