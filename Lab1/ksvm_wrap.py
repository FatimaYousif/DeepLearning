from sklearn.svm import SVC
from data import *

class KSVMWrap(SVC):

    # svm_c = cost
    # svm_gamma = RBF kernel width
    
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        super(KSVMWrap, self).__init__()
        self.X = X
        self.Y_ = Y_
        self.param_svm_c = param_svm_c
        self.param_svm_gamma = param_svm_gamma
        self.model = SVC(C=self.param_svm_c, kernel='rbf', gamma=self.param_svm_gamma)
        self.model.fit(self.X, self.Y_)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.decision_function(X)
    
    def support(self):
        # Indices of data chosen as support vectors
        return self.model.support_

def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
    """
    return model.get_scores(X)

def decfun(model):
    def classify(X):
      return eval(model, X)
    return classify

if __name__ == "__main__":
    np.random.seed(100)
    X, Y_ = sample_gmm_2d(6, 2, 1000)
    # X, Y_ = sample_gauss_2d( 2, 1000)
    
    model = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')
    Y = model.predict(X)


    accuracy, recall, precision = eval_perf_multi(Y, Y_)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))
    average_precision = eval_AP(Y)
    print("Average precision: {}".format(average_precision))

    # visualize the results, decicion surface
    decfun = decfun(model)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0)

    # SPECIAL NEEDS TO BE ADDED TO HIGHLIGHT THE SUPPORT VECTORS -----
    graph_data(X, Y_, Y, special=model.support())
    plt.show()