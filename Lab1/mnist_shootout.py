import torch
import torchvision
from pt_deep import *
import os
from sklearn.svm import SVC



# train w 
def early_stop(model, X, Y, iteration, lr, param_lambda, X_validation, Y_validation):
    fl = True
    for i in range(iteration):
        train(model, X, Y, 1, lr, param_lambda)
        probs = eval(model, X_validation.view(X_validation.shape[0], X_validation.shape[1]*X_validation.shape[2]))
        Y_true = np.argmax(probs, axis=1)
        accuracy, recall, precision = eval_perf_multi(Y_true, Y_validation)
        if i % 100 == 0:
            print(f'Iteration: {i}, loss:{model.loss:.3f}, accuracy: {accuracy:.3f}')
        # performance (ARP) stablized ~0.91
        if model.loss < 0.91 and fl:
            weights = model.weights
            biases = model.biases
            loss = model.loss
            fl = False
            break

    return weights, biases, loss

# training on minibatches
def train_mb(model, X, Y, iterations, lr, param_lambda, mini_batches=100, optimizer=optim.SGD):
    N = X.shape[0]
    stored_loss = []
    
    for i in range(iterations):
        shufled_list = torch.randperm(N)
        batch_size = int(N/mini_batches)
        for j in range(mini_batches):
            idx = shufled_list[j*batch_size:(j+1)*batch_size]
            X_mb = X[idx]
            Y_mb = Y[idx]
            train(model, X_mb, Y_mb, 1, lr, param_lambda, optimizer)
            stored_loss.append(model.loss.detach().numpy())
        if i % 100 == 0:
            print(f'Iteration: {i}, loss:{model.loss}')
    return stored_loss

# Make a function to train and evaluate linear SVM classifier using the module
# sklearn.svm.SVC by using one v one (OVO) SVM variant for multiclasss classification
def train_svm(X, Y):
    svm = SVC(kernel='linear', decision_function_shape='ovo')
    svm.fit(X, Y)
    Y_predicted = svm.predict(X)
    accuracy, recall, precision = eval_perf_multi(Y_predicted, Y)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

# Make a function to train and evaluate kernel SVM classifier using the module
# sklearn.svm.SVC by using one v one SVM variant for multiclasss classification
def train_kernel_svm(X, Y):
    svm = SVC(kernel='rbf', decision_function_shape='ovo')
    svm.fit(X, Y)
    Y_predicted = svm.predict(X)
    accuracy, recall, precision = eval_perf_multi(Y_predicted, Y)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

if __name__ == '__main__':
    # script_directory = os.path.dirname(os.path.abspath(__file__))

    dataset_root = 'FCNNs/mnist'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    # # Visualize random images from the training set
    # random_integers = [random.randint(0, 30000) for _ in range(200)]
    # for i in random_integers:
    #     plt.imshow(x_train[i].numpy(), cmap='gray')
    #     plt.show()


    # evaluate_random_model(x_train, y_train, x_test, y_test)

    N=x_train.shape[0]
    D=x_train.shape[1]*x_train.shape[2]
    C=y_train.max().add_(1).item()

    y_train_oh = class_to_onehot(y_train)

    N_validation = N//5
    shufled_data_list = torch.randperm(N)

    x_validation = x_train[shufled_data_list[:N_validation]]
    y_validation = y_train[shufled_data_list[:N_validation]]

    # # ptlr = PTDeep([D, C])
    ptlr = PTDeep([784, 10])
    # # ptlr = PTDeep([784, 100,10])
    

    # ---------- TODO : for WITHOUT TRAIN question------------------
    # Evaluate on training data
    # print("Evaluating on Training Set...")
    # probits_train = ptlr.forward(x_train.view(N, D))
    # loss_train = ptlr.get_loss(probits_train, torch.Tensor(y_train_oh))
    # probs_train = eval(ptlr, x_train.view(N, D))
    # preds_train = np.argmax(probs_train, axis=1)
    # accuracy_train, _, _ = eval_perf_multi(preds_train, y_train)
    # # print(f"Train Loss: {loss_train}")
    # print(f"Train Loss: {loss_train:.3f}, Train Accuracy: {accuracy_train * 100:.2f}%")

    # y_test_oh = class_to_onehot(y_test)

    # # Evaluate on testing data
    # print("Evaluating on Test Set...")
    # N_test = x_test.shape[0]
    # D_test = x_test.shape[1]*x_test.shape[2]
    # # probs = eval(ptlr, x_test.view(N_test, D_test))
    # probits_test = ptlr.forward(x_test.view(N_test, D_test))
    # loss_test = ptlr.get_loss(probits_test, torch.Tensor(y_test_oh))
    # probs_test = eval(ptlr, x_test.view(N_test, D_test))
    # preds_test = np.argmax(probs_test, axis=1)
    # accuracy_test, _, _ = eval_perf_multi(preds_test, y_test)
    # print(f"Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test * 100:.2f}%")

    # --------------------------------------------

    '''
    Training functions
    '''
    # # Train with full batch
    print('Training with full batch')
    stored_loss = train(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 6000, 0.2, 0.1)  #were 10k itr prev
    plt.plot(stored_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    # -------------------------------------
    # ---- TODO:2
    # Plot the images which contribute the most to the loss
    # image_indices = [indices[1] for indices in ptlr.most_loss_images]
    # for i in image_indices:
    #     plt.imshow(x_train[i].numpy(), cmap='gray')
    #     plt.show()
    # -------------------------------------
    

    # -------------------------------------
    # TODO Task 4 : VALIDATION - EARLY STOPPING
    # Train with early stopping
    # weights, biases, loss = early_stop(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 6000, 0.2, 0.1, x_validation, y_validation)
    # -------------------------------------

    # -------------Train with mini-batches
    # stored_loss = train_mb(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 2000, 0.1, 0.1, 100, optimizer=optim.SGD)
    # stored_loss = train_mb(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 2000, 1e-4, 0.1, 100, optimizer=optim.Adam)
    
    # ...adam with scheduler in train() of PTDeep for each mini batch---
    # stored_loss = train_mb(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 1000, 1e-4, 0.1, 100, optimizer=optim.Adam)

    # plt.plot(stored_loss)
    # plt.show()
    # ------------------------------------


    # last TODO:-------------------------------------
    # # Train using linear SVM
    # train_svm(x_train.view(N, D).numpy(), y_train.numpy())

    # # Train using kernel SVM
    # train_kernel_svm(x_train.view(N, D).numpy(), y_train.numpy())
    # ------------------------------------

    # generate computation graph
    # make_dot(ptlr.prob, params=ptlr.state_dict()).render("MNISTComputationGraph",
    #                                                     directory=script_directory, format="png", cleanup=True)

    # -------------------------------------
    # TODO ------ TASK 1.
    # Weights = ptlr.weights
    # for i, w in enumerate(Weights):         #W from each layer
    #     for i in range(w.size(1)):          #each OP class = 10
    #         weight = w[:, i].detach().view(28, 28).numpy()      #W of ith class -> conversions
    #         weight = (((weight - weight.min()) / (weight.max() - weight.min())) * 255.0).astype(np.uint8)
    #         plt.imshow(weight, cmap='gray')
    #         plt.title('Weights for class {}'.format(i))
    #         plt.show()
    # -------------------------------------

    # torch.save(ptlr.state_dict(), 'FCNNs/saved_weights/model_weights.pth')


    # print('Total number of parameters: ', count_params(ptlr))

    # # # Print evaluation metrics for the training set
    # # get probabilites on training data
    # print('Training set Metrics')
    # probs = eval(ptlr, x_train.view(N, D))
    # Y = np.argmax(probs, axis=1)
    # # print out the performance metric (precision and recall per class)
    # accuracy, recall, precision = eval_perf_multi(Y, y_train)
    # print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))



    # # Print evaluation metrics for the test set
    # N_test = x_test.shape[0]
    # D_test = x_test.shape[1]*x_test.shape[2]
    # probs = eval(ptlr, x_test.view(N_test, D_test))

    # # get probabilites on training data
    # print('Test set Metrics')
    # probs = eval(ptlr, x_test.view(N_test, D_test))
    # Y = np.argmax(probs, axis=1)
    # # print out the performance metric (precision and recall per class)
    # accuracy, recall, precision = eval_perf_multi(Y, y_test)
    # print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))


    # # Print evaluation metrics for the validation set
    # print('Validation set Metrics')
    # N_validation = x_validation.shape[0]
    # D_validation = x_validation.shape[1]*x_validation.shape[2]
    # probs = eval(ptlr, x_validation.view(N_validation, D_validation))
    # Y = np.argmax(probs, axis=1)
    # # print out the performance metric (precision and recall per class)
    # accuracy, recall, precision = eval_perf_multi(Y, y_validation)
    # print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))


    # # --- Print evaluation metrics for the early stopping
    # ---- TASK 4 
    # ptlr.weights = weights
    # ptlr.biases = biases

    # print('Early stopping Metrics')
    # probs = eval(ptlr, x_test.view(N_test, D_test))
    # Y = np.argmax(probs, axis=1)
    # # print out the performance metric (precision and recall per class)
    # accuracy, recall, precision = eval_perf_multi(Y, y_test)
    # print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))