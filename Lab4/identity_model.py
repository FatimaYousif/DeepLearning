import torch.nn as nn

# classification on the validation subset, but this time in the image space (not  --- metric embedding model)
# baseline model
# raw pixel (vs. learned embedding)
# just reshaping the input to 1D

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = img.view(img.shape[0], -1)
        return feats

