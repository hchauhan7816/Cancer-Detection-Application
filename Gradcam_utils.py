
import torch
import argparse
import torch.nn as nn
from torch.utils import data
from torchvision.models import densenet201
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class DenseNet(nn.Module):
    def __init__(self, model):
        super(DenseNet, self).__init__()
        self.densenet = model
        self.features_conv = self.densenet.features
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = self.densenet.classifier
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.global_avg_pool(x)
        x = x.view((1, 1024))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


class AlexNet(nn.Module):
    def __init__(self, model):
        super(AlexNet, self).__init__()
        self.alexnet = model
        self.features_conv = self.alexnet.features[:12]
        self.max_pool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.classifier = self.alexnet.classifier
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


def run_inference(model, dataloader, model_name="DenseNet"):
    if (model_name == "AlexNet"):
        ds = AlexNet(model)
    else:
        ds = DenseNet(model)
    ds.eval()
    img, _ = next(iter(dataloader))
    scores = ds(img)
    label = torch.argmax(scores)
    #print("label :: ", label);
    return ds, img, scores, label


def get_grad_cam(ds, img, scores, label, isAlexNet=False):
    scores[:, label].backward(retain_graph=True)
    gradients = ds.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = ds.get_activations(img).detach()

    if (isAlexNet):
        for i in range(256):
            activations[:, i, :, :] *= pooled_gradients[i]
    else:
        for i in range(1024):
            activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    # plt.matshow(heatmap.squeeze())
    return heatmap


def render_superimposition(root_dir, heatmap, image, isAlexNet=False):
    img = cv2.imread(os.path.join('images/hello/', image))
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    if(isAlexNet):
        cv2.imwrite(root_dir + '/images/alex_superimposed_' +
                    image, superimposed_img)
    else:
        cv2.imwrite(root_dir + '/images/superimposed_' +
                    image, superimposed_img)

    # cv2.imshow('output', superimposed_img)
