---
title: Batch Normalisation and Layer Normalisation
top: false
cover: false
toc: true
mathjax: true
date: 2021-09-17 23:41:52
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/17/2021-09-17-batch-norm-and-layer-norm/wallhaven-m9wjv9.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/17/2021-09-17-batch-norm-and-layer-norm/wallhaven-m9wjv9.jpg?raw=true
summary: The standardisation of inputs may be applied to input variables for the first hidden layer or to the activations from a hidden layer for deeper layers. In common, this normalisation technique is used on the inputs to the layer before or after the activation function in the previous layer. Using normalisation technique, in addition, can make the network more stable during training. In this articale, batch normalisation and layer normalisation will be compared.
categories: Deep Learning
tags:
	- Deep Learning
  - Neural Network
---

# Introduction

The main purpose of normalisation is to provide a uniform scale for numerical values. The standardisation of inputs may be applied to input variables for the first hidden layer or to the activations from a hidden layer for deeper layers. In common, this normalisation technique is used on the inputs to the layer before or after the activation function in the previous layer. Using normalisation technique, in addition, can make the network more stable during training. In this articale, batch normalisation (BN) and layer normalisation (LN) will be compared.

# Background

Yan LeCun emphasised the significance of normalising the inputs in his classic paper Effiecient BackProp in 1998. The use of normalisation to preprocess the inputs is a standard machine learning approach that is known to aid in faster convergence.

<figure>
  <img src="prepro1.jpeg" width=600>
  <figcaption style='text-align: center;'>Common data preprocessing pipeline. <strong>Left</strong>: Original toy, 2-dimensional input data. <strong>Middle</strong>: The data is zero-centered by subtracting the mean in each dimension. The data cloud is now centered around the origin. <strong>Right</strong>: Each dimension is additionally scaled by its standard deviation. The red lines indicate the extent of the data, and they are of unequal length in the middle, but of equal length on the right.</figcaption>
</figure>

# Batch Normalisation and Layer Normalisation

**Batch Normalisation**

LN is more commonly employed than BN in NLP, for example, transformer based models use LN instead of BN. What causes this to happen? In order to understand the reason behind this, we have to dig into the difference between BN and LN.

For neural networks that are trained in mini-batches, the batch normalisation layer is used. We partition the data into batches of a specified size before sending it over the network. For all of the data in the mini-batch, batch normalisation is applied to the neuron activation so that the mean output is close to 0 and the standard deviation is close to 1. It also includes two learning parameters, gamma and beta, which are optimised during training.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle z_{BN} = (\frac{z-\mu_{z}}{\sigma_{z}}) \cdot \gamma + \beta
    {% endmathjax %}
</div>

where {% mathjax %} z {% endmathjax %} is the neuron's output of the previous layer, {% mathjax %} \mu_{z} {% endmathjax %} is the mean of the neuron's output, and {% mathjax %} \sigma_{z} {% endmathjax %} is the standard deviation of the neuron's output. {% mathjax %} \gamma {% endmathjax %} and {% mathjax %} \beta {% endmathjax %} are the learning parameters of BN. Therefore, the outputs of BN over a layer will become having a distribution with a mean of {% mathjax %} \beta {% endmathjax %} and a standard deviation of {% mathjax %} \gamma {% endmathjax %}. These two parameters are learned over epochs.

**Layer Normalisation**

Layer normalisation is the next type of normalisation layer, and it overcomes the pitfalls of batch normalisation. The normalisation is done to the neuron for a single instance across all features, instead of batch-dependent. The mean activation is also close to 0, while the mean standard deviation is near to 1.

<figure>
  <img src="norm.png" width=600>
  <figcaption style='text-align: center;'><strong>Left</strong>: batch normalisation (BN). <strong>Right</strong>: layer normalisation (LN).</figcaption>
</figure>

In a case of NLP, assume that the dimension of word embedding is 10, the length of the sequence is 6, and the batch size is 8, as shown in the figure above.

* BN: fix every position of single token, and the shape of the matrix is (8, 10).
* LN: fix every single sequence, and the shape of the matrix is (6, 10).

The difference is, BN takes every vector of size 8 to scale, however, LN takes every vector of size 10 to scale. If we make a collection of text into a batch, the direction of BN is to operate on every token in each sentence. However, the language is quite complex, and any token may be placed in any position. Furthermore, the order of each token may not affect our understanding of the sentence. The BN is scaled for each position, which does not correspond to the NLP. On the other hand, LN scales it on every sentence, and LN normally operate on the dimension of embedding vector. It makes more sense using LN on NLP tasks instead of BN.

## Some Tips of Using BN

Generally, BN works well with most neural networks types, such as MLP, CNN, and RNN etc. There are quite a few tips of using BN properly.

First of all, BN is better used after the activation function if for s-shaped functions (TanH or Sigmoid), and is better used before the activation function if for non-Gaussian distributions (ReLU). In addition, using BN makes the network more stable during training process. This may require a lager learning rate to further speed up the learning process with no ill side effects. 

Second, BN offers some regularisation effect, so it is not a must to add dropout for regularisation. The reason is that when normalising the previous layer's output may become more noisy given the random dropping out of nodes during dropout procedure.

## Pros and Cons

The table below shows the advantages and disadvantages for BN and LN.

|| Pros | Cons |
| --- | --- | --- |
| BN | 1. Improves training time and accuracy. <br>2. Decrease the effect of weight initialisation. <br>3. Add a regularisation effect on the network. <br>4. Work better with FCN and CNN. <br>5. Stabilise and speed-up the training process. <br>6. Shrink internal covariate shift. | 1. Batch-dependent during training. <br>2. Does not work well with RNN. |
| LN | 1. Batch-independent during training. <br>2. Work better with RNN. | 1. May not produce good results with CNN. |

# Conclusion

Simply put, all the normalisation is calculated using the equation {% mathjax %} \hat{x} = \frac{x-mean(\bar{x})}{std(\bar{x})} {% endmathjax %}. The difference between BN and LN is they compute {% mathjax %} \bar{x} {% endmathjax %} in a different way. In BN, {% mathjax %} \bar{x} {% endmathjax %} is all of the summed inputs of single neuron on single batch. In LN, {% mathjax %} \bar{x} {% endmathjax %} is all of the summed inputs to the neurons in a layer on a single training case.

# References

1. https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
2. https://machinelearningknowledge.ai/keras-normalization-layers-explained-for-beginners-batch-normalization-vs-layer-normalization/
3. https://www.kdnuggets.com/2020/08/batch-normalization-deep-neural-networks.html
4. https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
5. https://zaffnet.github.io/batch-normalization
6. https://yichengsu.github.io/2019/12/pytorch-batchnorm-freeze/
7. https://www.quora.com/What-are-the-practical-differences-between-batch-normalization-and-layer-normalization-in-deep-neural-networks