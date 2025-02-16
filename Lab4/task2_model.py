import torch
import torch.nn as nn
import torch.nn.functional as F

# https://gombru.github.io/assets/ranking_loss/triplet_loss_faces.png
class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()

        # TODO - given
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k//2, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size

        # TODO -given
        self.conv1 = _BNReluConv(input_channels, emb_size, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = _BNReluConv(emb_size, emb_size, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = _BNReluConv(emb_size, emb_size, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # global avg pool = condense info
        self.g_avg = nn.AdaptiveAvgPool2d(1)

    # TODO
    def get_features(self, img):
        # returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        img = img.to(next(self.parameters()).device)  # Move input to model's device
  
        x = self.conv1(img)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.g_avg(x)
        x = x.view(x.size(0), -1)       # retain minibatch size
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        # TODO - euclidean dist = L2 norm 
        d_ap = torch.norm(a_x - p_x, p=2, dim=1)
        d_an = torch.norm(a_x - n_x, p=2, dim=1)

        # triplet loss --> m=1 (margin)
        loss = torch.clamp(d_ap - d_an + 1, min=0).mean()

        return loss