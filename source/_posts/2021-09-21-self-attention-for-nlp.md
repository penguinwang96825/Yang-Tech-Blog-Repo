---
title: Self-Attention for NLP
top: false
cover: false
toc: true
mathjax: true
date: 2021-09-21 23:56:28
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/21/2021-09-21-self-attention-for-nlp/castle.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/21/2021-09-21-self-attention-for-nlp/castle.jpg?raw=true
summary: In short, an attention-based model "focuses" on each element of the input (a word in a sentence or a different position in an image, etc.). "Focusing" means projecting different levels of attention so that the input elements are treated differently and each element of the input is weighted differently to influence the result; a non-attention model treats each element "equally".
categories: NLP
tags:
	- NLP
	- Deep Learning
	- Python
---

# Introduction

In short, an attention-based model "focuses" on each element of the input (a word in a sentence or a different position in an image, etc.). "Focusing" means projecting different levels of attention so that the input elements are treated differently and each element of the input is weighted differently to influence the result; a non-attention model treats each element "equally".

# Attention Mechanism

In natural language processing (NLP), the attention mechanism outperformed the encoder decoder-based neural machine translation system. As the name implies, the attention mechanism is essentially designed to mimic the way humans look at objects. For example, when looking at a picture, people will not only grasp the picture as a whole, but will also pay more attention to a particular part of the picture, such as the location of a table, or the category of product, etc. In the field of translation, whenever people translate a passage, they usually start with the sentence, but when reading the whole sentence, it is certainly necessary to focus on the information of the words themselves, as well as the information of the relationship between the words before and after and the information of the context. In NLP, if sentiment classification is to be performed, it will certainly involve words that express sentiment in a given sentence, including but not limited to "happy", "frustrated", "knackered" and so on. The other words in these sentences are contextual, not that they are useless, but that they do not play as big a role as the emotive keywords. Under the above description, the attention mechanism actually consists of two parts.

1. The attention mechanism needs to decide which part of the whole input needs more attention.
2. Feature extraction from key sections to get important information.

<figure>
  <img src="weights.png" width=400>
</figure>

Let's take a sentence for example. We have a context "you had me at hello", and we already get the embedding vectors for each token. As you can see in figure, the sequence length is 5 (five tokens), and the embedding dimension is 3 (size of vector). The embedding vector for each token in the input sequence is fixed and does not contain any contextualised information. The purpose of self attention is to get a new embedding vector for each token by calculating the dependencies between representations considering the contextual information (self-attention model allows inputs to interact with each other). Assume we want to calculate the contextualised embedding for "hello", so we use the formula below.

<figure>
  <img src="softmax.png" width=500>
</figure>

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    w_{5, 1}, w_{5, 2}, w_{5, 3}, w_{5, 4}, w_{5, 5} = softmax(e_{5}^{T}, [e_{1}, e_{2}, e_{3}, e_{4}, e_{5}])
    {% endmathjax %}
</div>

<br>

where {% mathjax %} e_{5} {% endmathjax %} is the embedding vector for the token "hello", and {% mathjax %} w_{i, j} {% endmathjax %} is the attention weight for the token. Extend it to all the tokens "you", "had", "me", "at", "hello" as below.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \begin{aligned}
    w_{1, 1}, w_{1, 2}, w_{1, 3}, w_{1, 4}, w_{1, 5} & = softmax(e_{1}^{T}, [e_{1}, e_{2}, e_{3}, e_{4}, e_{5}]) \\
    w_{2, 1}, w_{2, 2}, w_{2, 3}, w_{2, 4}, w_{2, 5} & = softmax(e_{2}^{T}, [e_{1}, e_{2}, e_{3}, e_{4}, e_{5}]) \\
    w_{3, 1}, w_{3, 2}, w_{3, 3}, w_{3, 4}, w_{3, 5} & = softmax(e_{3}^{T}, [e_{1}, e_{2}, e_{3}, e_{4}, e_{5}]) \\
    w_{4, 1}, w_{4, 2}, w_{4, 3}, w_{4, 4}, w_{4, 5} & = softmax(e_{4}^{T}, [e_{1}, e_{2}, e_{3}, e_{4}, e_{5}]) \\
    w_{5, 1}, w_{5, 2}, w_{5, 3}, w_{5, 4}, w_{5, 5} & = softmax(e_{5}^{T}, [e_{1}, e_{2}, e_{3}, e_{4}, e_{5}]) \\
    \end{aligned}
    {% endmathjax %}
</div>

<figure>
  <img src="average.png" width=500>
</figure>

After computing all the attention weights for the token "hello", we can get the weighted sum for the contextualised embedding vector. Since the weights {% mathjax %} w_{5, 1}, w_{5, 2}, w_{5, 3}, w_{5, 4}, w_{5, 5} {% endmathjax %} is the output of the softmax function, it can be sum to 1 (you can think of it as a probability distribution).

## Non-Parametric version of Self Attention.

```python
import torch
import torch.nn as nn


class NonparametricSelfAttention(nn.Module):
    """
    Examples
    --------
    >>> context = torch.Tensor([
            [
                [0.6, 0.2, 0.8], 
                [0.2, 0.3, 0.1], 
                [0.9, 0.1, 0.8], 
                [0.4, 0.1, 0.4], 
                [0.4, 0.1, 0.6]
            ]
        ])
    >>> context_, attention_weights = NonparametricSelfAttention(3)(context)
    >>> print("Input: ", context_.shape)
    Input:  torch.Size([1, 5, 3])
    >>> print("Output: ", context_.shape)
    Output:  torch.Size([1, 5, 3])
    """
    def __init__(self, dimensions):
        super(NonparametricSelfAttention, self).__init__()
        self.dimensions = dimensions
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, context, return_weights=True):
        """
        context: [sequence_length, embedding_dimension]
        """
        attention_scores  = torch.bmm(context, context.transpose(1, 2))
        attention_weights = self.softmax(attention_scores )
        context_ = torch.bmm(attention_weights, context)
        if return_weights:
            return context_ , attention_weights
        return context_ 

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

## Parametric version of Self Attention.

```python
import math
import torch
import torch.nn as nn


class ParametricSelfAttention(nn.Module):
    """
    Examples
    --------
    >>> context = torch.Tensor([
            [
                [0.6, 0.2, 0.8], 
                [0.2, 0.3, 0.1], 
                [0.9, 0.1, 0.8], 
                [0.4, 0.1, 0.4], 
                [0.4, 0.1, 0.6]
            ]
        ])
    >>> context_, attention_weights = ParametricSelfAttention(3)(context)
    >>> print("Input: ", context_.shape)
    Input:  torch.Size([1, 5, 3])
    >>> print("Output: ", context_.shape)
    Output:  torch.Size([1, 5, 3])
    """
    def __init__(self, dimensions):
        super(ParametricSelfAttention, self).__init__()
        self.dimensions = dimensions
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.linear_q_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_k_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_v_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions, dimensions, bias=False)

    def forward(self, context, return_weights=True):
        """
        context: [sequence_length, embedding_dimension]
        """
        context_q = self.linear_q_in(context)
        context_k = self.linear_k_in(context)
        context_v = self.linear_v_in(context)

        attention_scores  = torch.bmm(context_q, context_k.transpose(1, 2))
        attention_weights = self.softmax(attention_scores / math.sqrt(self.dimensions))
        context_ = torch.bmm(attention_weights, context_v)
        context_ = self.tanh(context_)
        context_ = self.linear_out(context_)
        if return_weights:
            return context_ , attention_weights
        return context_ 

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

# Conclusions

In context-aware encoding for learning long-range dependencies, self-attention was utilised to replace RNN (Vaswani et al. 2017). The length of the paths along which the forward and backward signals move in the network affects the ability to learn long-range relationships.

# References

1. https://zhuanlan.zhihu.com/p/56501461
2. https://github.com/Jianlong-Fu/Recurrent-Attention-CNN