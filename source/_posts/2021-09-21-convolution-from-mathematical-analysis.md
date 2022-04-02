---
title: Convolution from Mathematical Analysis
top: false
cover: false
toc: true
mathjax: true
date: 2021-09-21 11:14:10
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/21/2021-09-21-convolution-from-mathematical-analysis/wallhaven-y8v7zx.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/21/2021-09-21-convolution-from-mathematical-analysis/wallhaven-y8v7zx.jpg?raw=true
summary: Convolution is a mathematical operation on two functions (f and g) that creates a third function (f * g) that expresses how the shape of one is modified by the other in mathematics (specifically, functional analysis). If one of the functions participating in the fold is considered to be the indicator function of the interval, the fold can also be considered to be a 'sliding average' promotion. The idea of applying the convolutional operation to image data is not novel or specific to convolutional neural networks. A convolution is simply the application of a filter to an input that results in an activation. In computer vision, it's a common technique.
categories: Deep Learning
tags:
	- Python
	- Deep Learning
	- CNN
---

# Introduction

Convolution is a mathematical operation on two functions ({% mathjax %} f {% endmathjax %} and {% mathjax %} g {% endmathjax %}) that creates a third function ({% mathjax %} f * g {% endmathjax %}) that expresses how the shape of one is modified by the other in mathematics (specifically, functional analysis). If one of the functions participating in the fold is considered to be the indicator function of the interval, the fold can also be considered to be a 'sliding average' promotion. The idea of applying the convolutional operation to image data is not novel or specific to convolutional neural networks. A convolution is simply the application of a filter to an input that results in an activation. In computer vision, it's a common technique.

# Definition

The convolution of {% mathjax %} f {% endmathjax %} and {% mathjax %} g {% endmathjax %} is written {% mathjax %} f * g {% endmathjax %}, denoting the operator with the symbol `*`. It is defined as the integral of the product of the two functions after one is reversed and shifted. As such, it is a particular kind of integral transform

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle (f*g)(t) = \int_{\mathbb{R}} f(\tau) g(t-\tau) d\tau
    {% endmathjax %}
</div>

## Visual Explanation

In the below example, the red-colored {% mathjax %} g(\tau) {% endmathjax %} is an even function, and {% mathjax %} f(\tau) {% endmathjax %} in blue is fixed for some value of parameter {% mathjax %} t {% endmathjax %}. The amount of yellow is the area of the product {% mathjax %} f(\tau) \cdot g(t-\tau) {% endmathjax %}, computed by the convolution integral. The movement is created by continuously changing {% mathjax %} t {% endmathjax %} and recomputing the integral. The result in black is a function of {% mathjax %} t {% endmathjax %}.

<figure>
  <img src="convolution_of_box_signal_with_itself.gif" width=600>
  <img src="convolution_of_spiky_function_with_box.gif" width=600>
  <figcaption style='text-align: center;'>Source from https://en.wikipedia.org/wiki/Convolution</figcaption>
</figure>

## Algebraic Properties

The convolution defines a product on the linear space of integrable functions.

1. Commutativity: {% mathjax %} f * g = g * f {% endmathjax %}
2. Associativity: {% mathjax %} f * (g * h) = (f * g) * h {% endmathjax %}
3. Distributivity: {% mathjax %} f * (g + h) = (f * g) + (f * h) {% endmathjax %}

## Computation with the Image and Kernel

Detkov and Nikita (2020) wrote the convolution from scratch using pure python and numpy, which can be checked out in their [github](https://github.com/detkov/Convolution-From-Scratch). A more precise tutorial of convolution of input signal in 2D spatial can be seen in Song's [article](http://www.songho.ca/dsp/convolution/convolution2d_example.html)

<figure>
  <img src="convolution_process.gif" width=600>
  <figcaption style='text-align: center;'>Apply convolution with {% mathjax %}stride = (2, 1){% endmathjax %} and {% mathjax %}dilation = (1, 2){% endmathjax %} without padding. Source from https://github.com/detkov/Convolution-From-Scratch</figcaption>
</figure>

## Effect on Images

Computer vision is common in daily life whether we are aware of it or not. For starters, we see filtered photographs in our social media feeds, news stories, publications, and novels. Image processing can be divided into two categories: image filtering and image warping. Image filtering alters the range (or pixel values) of an image, causing the image's colours to change without affecting the pixel locations, whereas image warping alters the domain (or pixel positions), causing points to be mapped to other points without changing the colours. 

We'll take a closer look at picture filtering. Filters are used to change or enhance visual features and extract valuable information from images such as edges, corners, and blobs. Here are some instances of how applying filters to photos can improve their visual attractiveness.Four images are compared below.

<figure>
  <img src="ping.png" width=600>
</figure>

# Convolutional Neural Networks

Basically, in neural networks, the convolution layer uses filters that perform convolution operations as it is scanning the input {% mathjax %} I {% endmathjax %} with respect to its dimensions. Its hyperparameters include the filter size {% mathjax %} F {% endmathjax %} and stride {% mathjax %} S {% endmathjax %}. The resulting output {% mathjax %} O {% endmathjax %} is called feature map or activation map.

Generally, when the API is running tensor, it is usually a batch of data that is being executed. Suppose there are 100 samples of data, and each sample of data is a {% mathjax %}10 \times 10{% endmathjax %} colour picture (so each piece of data is 3 channels, R, G and B respectively), then the size of the input data is {% mathjax %}100 \times 10 \times 10 \times 10 \times 3{% endmathjax %}, and the interpretation method is {% mathjax %}batch \times height \times width \times channel{% endmathjax %}.

Without further ado, let's look at a diagram. The following diagram shows the general convolution calculation. In this example, the shape of the image is {% mathjax %}4 \times 4 \times 3 \times 3{% endmathjax %}. There are two {% mathjax %}3 \times 3 \times 3{% endmathjax %} kernel maps being set in first convolution layer, so there'll be two feature maps for the output. In second convolution layer, there are three {% mathjax %}3 \times 3 \times 2{% endmathjax %} kernel maps being set, so there'll be three feature maps for the output. **So you can see that the number of channels after convolution is determined by the number of kernel maps you set.**

<figure>
  <img src="conv.png" width=600>
  <figcaption style='text-align: center;'>Source from Tommy's blog{% mathjax %}^{[7]}{% endmathjax %}.</figcaption>
</figure>

You must be wondering why the above diagram showing that after applying the convolution operation, why isn't the feature map being smaller? If you have played with the deep learning API, there are usually two parameters (strides, padding) that can be tuned in addition to the basic input and filter (kernel map) part of the convolution calculation. In most cases, we would like to let the input map not be affected by the size of the kernel map. The formula for calculating the length and width of a new feature map after convolution operation is as follows.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle W_{new} = floor(\frac{W_{old} + 2 \times P - F}{S}) + 1
    {% endmathjax %}
</div>

where we assume that the width and height {% mathjax %} W {% endmathjax %} of the image are the same. {% mathjax %} F {% endmathjax %} stands for filter (kernel map) size, {% mathjax %} S {% endmathjax %} stands for pace length of kernel map when moving, and {% mathjax %} P {% endmathjax %} stands for padding size.

If you are not sure about the output size after computing convolution operation, you can calculate it on Edward's [Convolution Visualizer](https://ezyang.github.io/convolution-visualizer/index.html). This interactive visualisation shows how different convolution settings affect the input, weight, and output matrices' forms and data dependencies, or you can calculate it with the formula. If the input has non-equal width and height, just simply apply the formula separately on the width and the height.

PyTorch does not support same padding the way Keras does, but still we can manage it easily using explicit padding before passing the tensor to convolution layer. Here is a very simple `Conv2d` layer with `same` padding for reference. It only support square kernels and `stride=1`, `dilation=1`, `groups=1`. 

```python
import torch
import torch.nn as nn


class Conv2dSame(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 bias=True, 
                 padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb ,ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
        
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    input_ = torch.rand((16, 1, 32, 100)).float()
    output_ = Conv2dSame(in_channels=1, out_channels=3, kernel_size=4)(input_)
    print(input_.shape)
    print(output_.shape)
```

# CNN Case Studies

## LeNet-5

LeNet-5 is Yann LeCun's convolutional network designed for handwritten and machine-printed character recognition in the year 1998. In their research [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), LeNet-5 is popular due to its simple and straightforward architecture (multi-layer convolution neural network).

<figure>
  <img src="lenet5.png" width=600>
  <figcaption style='text-align: center;'>Final architecture of the Lenet-5 model. The input to the model is a grayscale image. It has 3 convolution layers, two average pooling layers, and two fully connected layers with a softmax classifier.</figcaption>
</figure>

```python
import torch
import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    Gradient-Based Learning Applied to Document Recognition
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, kernel_size=5)), 
            ('activate1', nn.Tanh()), 
            ('pool1', nn.MaxPool2d(kernel_size=2)), 
            ('conv2', nn.Conv2d(6, 16, kernel_size=5)), 
            ('activate2', nn.Tanh()), 
            ('pool2', nn.MaxPool2d(kernel_size=2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16*5*5, 120)), 
            ('activate1', nn.Tanh()), 
            ('fc2', nn.Linear(120, 84)), 
            ('activate2', nn.Tanh()), 
            ('fc3', nn.Linear(84, 10))
        ]))

    def forward(self, x):
        """
        x: torch.tensor
            A torch tensor of shape [batch, 1, 32, 32]
        """
        x = self.conv(x)
        x = Flatten()(x)
        x = self.fc(x)
        return x

    
class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
if __name__ == "__main__":
    input_ = torch.rand((16, 1, 32, 32)).float()
    output_ = LeNet5()(input_)
    print(input_.shape)
    print(output_.shape)
```

## AlexNet

AlexNet, proposed by Alex Krizhevsky, is one of the popular variants of the convolutional neural network and used as a deep learning framework. In Alex's [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), AlexNet model has a very good performance when it is used as a transfer learning framework.

<figure>
  <img src="alexnet.png" width=600>
</figure>

```python
import torch
import torch.nn as nn
from collections import OrderedDict


class AlexNet(nn.Module):
	"""
	ImageNet Classification with Deep Convolutional Neural Networks
	"""
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)), 
            ('relu1', nn.ReLU(inplace=True)), 
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)), 
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)), 
            ('relu2', nn.ReLU(inplace=True)), 
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)), 
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)), 
            ('relu3', nn.ReLU(inplace=True)), 
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)), 
            ('relu4', nn.ReLU(inplace=True)), 
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)), 
            ('relu5', nn.ReLU(inplace=True)), 
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout()),
            ('fc1', nn.Linear(256*6*6, 4096)),
            ('relu1', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout()),
            ('fc2', nn.Linear(4096, 4096)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(4096, num_classes)),
        ]))

    def forward(self, x):
        """
        x: torch.Tensor
            A torch tensor of shape [batch, 3, 224, 224]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
if __name__ == "__main__":
    input_ = torch.rand((16, 3, 224, 224)).float()
    output_ = AlexNet()(input_)
    print(input_.shape)
    print(output_.shape)
```

## VGG

There are six models available in [VGG](https://arxiv.org/pdf/1409.1556.pdf). VGG mainly has three parts: convolution, pooling, and fully connected layers. VGG resulted in a significant increase in accuracy as well as a significant increase in speed. This was mostly due to the model's increased depth and the incorporation of pretrained models. However, this model suffers from the vanishing gradient problem, which is a significant drawback. The ResNet architecture was used to tackle the vanishing gradient problem.

<figure>
  <img src="vgg.jpg" width=400>
</figure>

```python
import math
import torch
import torch.nn as nn
from collections import OrderedDict


class VGG(nn.Module):
    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    def __init__(self, structure, dropout=0.5, batch_norm=True, num_classes=1000):
        """
        Examples
        --------
        >>> input_ = torch.rand((16, 3, 224, 224)).float()
        >>> VGG11 = VGG('A')
        >>> VGG13 = VGG('B')
        >>> VGG16 = VGG('D')
        >>> VGG19 = VGG('E')
        """
        super(VGG, self).__init__()
        self.features = self.make_layers(self.cfg[structure], batch_norm=batch_norm)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512*7*7, 4096)),
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc2', nn.Linear(4096, 4096)),
            ('relu2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(p=dropout)),
            ('fc3', nn.Linear(4096, num_classes)),
        ]))
        self._initialize_weights()

    def forward(self, x):
        """
        x: torch.Tensor
            A torch tensor of shape [batch, 3, 224, 224]
        """
        x = self.features(x)
        x = Flatten()(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    
if __name__ == "__main__":
    input_ = torch.rand((16, 3, 224, 224)).float()
    print(input_.shape)
    output_ = VGG('A')(input_) # VGG11
    print(output_.shape)
    output_ = VGG('B')(input_) # VGG13
    print(output_.shape)
    output_ = VGG('D')(input_) # VGG16
    print(output_.shape)
    output_ = VGG('E')(input_) # VGG19
    print(output_.shape)
```

## ResNet

By integrating skip connections, the ResNet proposed in "Deep Residual Learning for Image Recognition" overcame the degradation problem. According to the authors, optimising the network with skip connections is easier than optimising the original, unreferenced mapping.

<figure>
  <img src="resnet.png" width=300>
</figure>

Representation of residual networks with 18, 34, 50, 101, and 152 layers.

<figure>
  <img src="resnet-table.png" width=600>
</figure>

```python
import torch
import torch.nn as nn
from collections import OrderedDict


class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_layers, downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "Not a valid architecture."
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    

class ResNet(nn.Module):
    
    def __init__(self, in_channels=3, num_layers=18, block=Block, num_classes=1000):
        assert num_layers in [18, 34, 50, 101, 152], "Not a valid architecture."
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def forward(self, x):
        """
        x: torch.Tensor
            A torch tensor of shape [batch, 3, 224, 224]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = Flatten()(x)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []
        downsample = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride)), 
            ('bn', nn.BatchNorm2d(intermediate_channels*self.expansion))
        ]))
        layers.append(block(self.in_channels, intermediate_channels, num_layers, downsample, stride))
        self.in_channels = intermediate_channels * self.expansion
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels, num_layers))
        return nn.Sequential(*layers)
    
    
if __name__ == "__main__":
    input_ = torch.rand((16, 3, 224, 224)).float()
    print(input_.shape)
    output_ = ResNet(num_layers=18)(input_)
    print(output_.shape)
    output_ = ResNet(num_layers=34)(input_)
    print(output_.shape)
    output_ = ResNet(num_layers=50)(input_)
    print(output_.shape)
    output_ = ResNet(num_layers=101)(input_)
    print(output_.shape)
    output_ = ResNet(num_layers=152)(input_)
    print(output_.shape)
```

# Conclusions

Lenet-5 is one of the earliest pre-trained models, which promoted the event of deep learning, proposed by Yann LeCun and others in the year 1998. Alexnet and VGG are pretty much the same concept, but VGG is deeper and has more parameters, as well has using only {% mathjax %}3 \times 3{% endmathjax %} filters. Resnets are a kind of CNNs called Residual Networks. They are very deep compared to Alexnet and VGG, and Resnet50 refers to a 50 layers Resnet. Resnet introduced residual connections between layers, meaning that the output of a layer is a convolution of its input plus its input. Moreover, layers in a Resnet also use Batch Normalization, which has also been incorporated to VGG.

# References

1. https://en.wikipedia.org/wiki/Convolution
2. https://machinelearningmastery.com/
3. https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381convolutional-layers-for-deep-learning-neural-networks/
4. https://github.com/detkov/Convolution-From-Scratch
5. https://en.wikipedia.org/wiki/Kernel_(image_processing)
6. https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html
7. https://datascience.stackexchange.com/questions/73929/what-are-advantages-or-disadvantages-of-training-deep-learning-model-from-scratc
8. https://github.com/vdumoulin/conv_arithmetic
9. https://github.com/pytorch/pytorch/issues/3867#issuecomment-407663012
10. https://chsasank.com/vision/_modules/torchvision/models/vgg.html
11. Tommy Huang, Convolutional neural network (CNN): what 1×1 convolutional computing is doing [[article](https://chih-sheng-huang821.medium.com/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-convolutional-neural-network-cnn-1-1%E5%8D%B7%E7%A9%8D%E8%A8%88%E7%AE%97%E5%9C%A8%E5%81%9A%E4%BB%80%E9%BA%BC-7d7ebfe34b8)]