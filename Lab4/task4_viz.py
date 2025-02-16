import numpy as np
import torch
from task1_loading import MNISTMetricDataset
from task2_model import SimpleMetricEmbedding
from matplotlib import pyplot as plt

# OP
# The feature space = better rep for classification = distinct clusters (of classes).
# The raw image space = less structured = high overlap bw classes.

def get_colormap():
    # cityscapes colormap for first 10 classes
    colormap = np.zeros((10, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    return colormap

def plot_with_labels(X, Y, colormap, title, path):
    plt.figure()
    for i in range(10):
        mask = Y == i
        plt.scatter(X[mask, 0], X[mask, 1], color=colormap[i] / 255., s=5, label=str(i))
    plt.legend()
    plt.title(title)
    plt.savefig(path)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")
    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size).to(device)
    
    # TODO - LOAD TRAINED PARAMS

    # all digits
    # model_path = 'Lab4/model_params/model_2.pth'
    
    # without 0
    model_path = 'Lab4/model_params/reduced_model_2.pth'
    
    model.load_state_dict(torch.load(model_path, map_location=device))    
    
    colormap = get_colormap()
    mnist_download_root = "Lab1/FCNNs/mnist"
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    X = ds_test.images
    Y = ds_test.targets
    print("Fitting PCA directly from images...")
    test_img_rep2d = torch.pca_lowrank(ds_test.images.view(-1, 28 * 28), 2)[0]
    plot_with_labels(test_img_rep2d, Y, colormap, "PCA from Images", path='Lab4/outputs/original_reduced_classes_0.png')
    
    print("Fitting PCA from feature representation")
    with torch.no_grad():
        model.eval()
        test_rep = model.get_features(X.unsqueeze(1).to(device))
        # test_rep2d = torch.pca_lowrank(test_rep, 2)[0]
        test_rep2d = torch.pca_lowrank(test_rep, 2)[0].cpu().numpy()
        plot_with_labels(test_rep2d, Y, colormap, "PCA from Feature Representation", path='Lab4/outputs/feature_reduced_classes_0.png')