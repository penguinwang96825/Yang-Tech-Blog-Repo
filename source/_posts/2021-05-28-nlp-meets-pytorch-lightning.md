---
title: NLP Meets PyTorch Lightning
top: false
cover: false
toc: true
mathjax: true
date: 2021-05-28 01:33:55
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/28/2021-05-28-nlp-meets-pytorch-lightning/wallhaven-e78j3l.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/28/2021-05-28-nlp-meets-pytorch-lightning/wallhaven-e78j3l.jpg?raw=true
summary: PyTorch Lightning is a Python package that provides a high-level interface for PyTorch, a well-known deep learning framework. It's a fast, lightweight framework that organises PyTorch code to separate research and engineering, making deep learning experiments easier to comprehend and reproduce.
categories: NLP
tags:
	- Python
	- PyTorch
	- Text
---

# Introduction

PyTorch Lightning is a Python package that provides a high-level interface for PyTorch, a well-known deep learning framework. It's a fast, lightweight framework that organises PyTorch code to separate research and engineering, making deep learning experiments easier to comprehend and reproduce.

This guide will walk you through the core pieces of PyTorch Lightning. I'll accomplish the following:
- Implement an text classifier.
- Use inheritance to implement a model.

# Load Data

As usual, load in the package we need.

```python
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
```

The [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database) is a collection of complaints about consumer financial products and services that we sent to companies for response. Complaints are published after the company responds, confirming a commercial relationship with the consumer, or after 15 days, whichever comes first. Complaints referred to other regulators, such as complaints about depository institutions with less than $10 billion in assets, are not published in the Consumer Complaint Database. The database generally updates daily.

In this article, I will only use three classes in this data, which are "Money transfers", "Prepaid card", and "Payday loan", to be more quick to implement.

```python
def load_dataframe():
    df = pd.read_csv(r"C:\Users\Yang\Desktop\Dissertation\data\test\complaints.csv")
    df = df[["Consumer complaint narrative", "Product"]]
    df.columns = ["text", "label"]
    df = df[(df["label"]=="Money transfers") | (df["label"]=="Prepaid card") | (df["label"]=="Payday loan")]
    df = df.dropna()
    df = df.reset_index(drop=True)
    df["text"] = df["text"].map(str)
    df["text"] = df["text"].apply(clean_text)
    
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    classes = le.classes_

    return df, y
```

## Construct PyTorch Dataset and Dataloader

```python
class TextDataset(torch.utils.data.Dataset):
    
    def __init__(self, input_ids, labels):
        self.X = input_ids
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return [X, y]

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False, num_workers=0):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True, 
            num_workers=num_workers)
```

---

```python
batch_size = 16
num_workers = 16

df, y = load_dataframe()
tokeniser = AutoTokenizer.from_pretrained("bert-base-uncased")
input_ids = tokeniser.batch_encode_plus(df["text"].tolist(), 
                                        padding=True, 
                                        max_length=128, 
                                        add_special_tokens=True, 
                                        return_tensors="pt", 
                                        return_token_type_ids=False, 
                                        return_attention_mask=False)

X_train, X_valid, y_train, y_valid = train_test_split(input_ids["input_ids"].numpy(), 
                                                      y, 
                                                      test_size=0.3, 
                                                      random_state=914, 
                                                      stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, 
                                                    y_valid, 
                                                    test_size=0.5, 
                                                    random_state=914, 
                                                    stratify=y_valid)

train_dataset = TextDataset(X_train, y_train)
valid_dataset = TextDataset(X_valid, y_valid)
test_dataset = TextDataset(X_test, y_test)

train_dataloader = train_dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)
valid_dataloader = valid_dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)
test_dataloader = test_dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)
```

# Model

This is the core code of PyTorch Lightning to run a multi-class classification task, the only thing we need to do is to change the `forward()` method.

```python
class MultiClass(pl.LightningModule):
    """
    Multi-class Classification
    """
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        
        # Set our learning rate
        self.learning_rate = learning_rate
        
        # Create loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict(self, test_dataloader):
        """Prediction step."""
        # Set model to eval mode
        self.eval()
        y_probs = []

        # Iterate over val batches
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                # Forward pass with inputs
                y_prob = self(x)
                # Store outputs
                y_probs.extend(y_prob.cpu())
        return self.softmax(np.vstack(y_probs))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        exp_ = np.exp(x - np.max(x))
        return exp_ / exp_.sum(axis=0)
```

In this tutorial, I will use a simple but powerful baseline, which is TextCNN. TextCNN is a convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. On numerous benchmarks, a simple CNN with little hyperparameter adjustment and static vectors provides outstanding performance.

```python
class TextCNN(MultiClass):
    """
    https://finisky.github.io/2020/07/03/textcnnmvp/
    """
    def __init__(self, embed_num, embed_dim, class_num, kernel_num=100, kernel_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()

        Ci = 1
        Co = kernel_num

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        x = self.embed(x)                                         # (N, token_num, embed_dim)
        x = x.unsqueeze(1)                                        # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]   # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]    # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1)                                       # (N, Co * len(kernel_sizes))
        x = self.dropout(x)                                       # (N, Co * len(kernel_sizes))
        logit = self.fc(x)                                        # (N, class_num)
        return logit

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)
```

## Run with Trainer

To use the lightning trainer simply:
- init your LightningModule and datasets
- init lightning trainer
- call `trainer.fit()`

```python
model = TextCNN(len(tokenizer.vocab), 300, 3, 100, [3, 4, 5], 0.1)
model.apply(init_weights)
trainer = pl.Trainer(gpus=1, fast_dev_run=False, max_epochs=50)
trainer.fit(model, train_dataloader, valid_dataloader)
```

Let's see how it performs.

```python
test_probs = model.predict(test_dataloader)
test_preds = np.argmax(test_probs, axis=1)

print(metrics.f1_score(test_preds, y_test, average="weighted"))
print(metrics.accuracy_score(test_preds, y_test))
```

This model achieves accuracy score of 93.9561% and f1 score of 94.0341% without any hyperparmeter tuning. It's quite promising.

Put it all together.

```python
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics


def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')
    text = text.strip()
    return text


def load_dataframe():
    df = pd.read_csv(r"C:\Users\Yang\Desktop\Dissertation\data\test\complaints.csv")
    df = df[["Consumer complaint narrative", "Product"]]
    df.columns = ["text", "label"]
    df = df[(df["label"]=="Money transfers") | (df["label"]=="Prepaid card") | (df["label"]=="Payday loan")]
    df = df.dropna()
    df = df.reset_index(drop=True)
    df["text"] = df["text"].map(str)
    df["text"] = df["text"].apply(clean_text)
    
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    classes = le.classes_

    return df, y
    

class TextDataset(torch.utils.data.Dataset):
    
    def __init__(self, input_ids, labels):
        self.X = input_ids
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return [X, y]

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False, num_workers=0):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True, 
            num_workers=num_workers)


class MultiClass(pl.LightningModule):
    """
    Multi-class Classification
    """
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        
        # Set our learning rate
        self.learning_rate = learning_rate
        
        # Create loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict(self, test_dataloader):
        """Prediction step."""
        # Set model to eval mode
        self.eval()
        y_probs = []

        # Iterate over val batches
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                # Forward pass with inputs
                y_prob = self(x)
                # Store outputs
                y_probs.extend(y_prob.cpu())
        return self.softmax(np.vstack(y_probs))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        exp_ = np.exp(x - np.max(x))
        return exp_ / exp_.sum(axis=0)


class TextCNN(MultiClass):
    """
    https://finisky.github.io/2020/07/03/textcnnmvp/
    """
    def __init__(self, embed_num, embed_dim, class_num, kernel_num=100, kernel_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()

        Ci = 1
        Co = kernel_num

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        x = self.embed(x)                                         # (N, token_num, embed_dim)
        x = x.unsqueeze(1)                                        # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]   # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]    # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1)                                       # (N, Co * len(kernel_sizes))
        x = self.dropout(x)                                       # (N, Co * len(kernel_sizes))
        logit = self.fc(x)                                        # (N, class_num)
        return logit


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


def run():
    batch_size = 16
    num_workers = 16

    df, y = load_dataframe()
    tokeniser = AutoTokenizer.from_pretrained("bert-base-uncased")
    input_ids = tokeniser.batch_encode_plus(df["text"].tolist(), 
                                            padding=True, 
                                            max_length=128, 
                                            add_special_tokens=True, 
                                            return_tensors="pt", 
                                            return_token_type_ids=False, 
                                            return_attention_mask=False)

    X_train, X_valid, y_train, y_valid = train_test_split(input_ids["input_ids"].numpy(), 
                                                          y, 
                                                          test_size=0.3, 
                                                          random_state=914, 
                                                          stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid, 
                                                        y_valid, 
                                                        test_size=0.5, 
                                                        random_state=914, 
                                                        stratify=y_valid)

    train_dataset = TextDataset(X_train, y_train)
    valid_dataset = TextDataset(X_valid, y_valid)
    test_dataset = TextDataset(X_test, y_test)

    train_dataloader = train_dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = valid_dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)
    test_dataloader = test_dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)

    model = TextCNN(len(tokeniser.vocab), 300, 3, 100, [3, 4, 5], 0.1)
    model.apply(init_weights)
    trainer = pl.Trainer(gpus=1, fast_dev_run=False, max_epochs=10)
    trainer.fit(model, train_dataloader, valid_dataloader)

    test_probs = model.predict(test_dataloader)
    test_preds = np.argmax(test_probs, axis=1)

    print(metrics.f1_score(test_preds, y_test, average="weighted"))
    print(metrics.accuracy_score(test_preds, y_test))


if __name__ == "__main__":
    run()
```

# Conclusion

In this blog post, I've show you how to use PyTorch Lightning to train any text model on any dataset, using advanced performance features of Lightning. Cheers!

# References

1. https://devblog.pytorchlightning.ai/training-transformers-at-scale-with-pytorch-lightning-e1cb25f6db29
2. https://degravek.github.io/project-pages/project1/2017/04/28/New-Notebook/
3. https://github.com/delldu/TextCNN
4. https://www.kaggle.com/thewillmundy/cassava-leaf-classification-with-pytorch-lightning
5. https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17