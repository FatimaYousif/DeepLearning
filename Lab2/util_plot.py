import os
import math
import torch
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

# --GIVEN---
def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)

def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.clone().data.cpu().numpy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    img = ski.img_as_ubyte(img)
    ski.io.imsave(os.path.join(save_dir, filename), img)

def draw_image(img, mean, std):
    img = img.transpose(1, 2, 0)
    img *= std
    img += mean
    img = img.astype(np.uint8)
    return img

# Visualize 20 incorrectly classified images with the largest loss and output the correct class and the top 3 predicted classes
def plot_images(save_dir, dataset, indexes, probs, mean, std):
    plot, axis = plt.subplots(2, 10, figsize=(20, 6))
    i = 0
    for row in axis:
        for col in row:
            index = indexes[i]
            image = draw_image(dataset[index][0].cpu().numpy(), mean, std)
            y = dataset[index][1].cpu().numpy()
            col.imshow(image)
            col.axis('off')
            top_3 = np.argsort(probs[i])[-3:][::-1]
            col.set_title(f'Predicted: {top_3}, True: {y}', fontsize=8)
            i += 1
    plt.savefig(os.path.join(save_dir, 'biggest_losses.png'))