---
title: Sentiment Analysis for KKBOX
date: 2019-07-10 09:25:00
author: Yang Wang
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2019/07/10/2019-07-10-sentiment-analysis-for-kkbox/piano-1638633.png?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2019/07/10/2019-07-10-sentiment-analysis-for-kkbox/john-matychuk.jpg?raw=true
summary: This sentiment classification task is based on reviews data of UtaPass and KKBOX from Google Play platform. As a KKStreamer at KKBOX, I become more interested in Natural Language Processing, especially text classification. First, I start crawling the text data using web crawler technique, namely BeautifulSoup and Selenium. Second, I develop several different neural network architectures, including simple RNN, LSTM, GRU, and CNN, to name but a few, to detect the polarity of reviews from customers.
categories: NLP
tags:
  - Python
  - NLP
  - KKBOX
  - UtaPass
---

# Sentiment Classification for UtaPass & KKBOX Reviews

Text classification for reviews of UtaPass & KKBOX using different deep learning models.

## Introduction

This sentiment classification task is based on reviews data of UtaPass and KKBOX from Google Play platform. As a KKStreamer at KKBOX, I become more interested in Natural Language Processing, especially text classification. First, I start crawling the text data using web crawler technique, namely BeautifulSoup and Selenium. Second, I develop several different neural network architectures, including simple RNN, LSTM, GRU, and CNN, to name but a few, to detect the polarity of reviews from customers.

## Data Source

1. [UtaPass](https://play.google.com/store/apps/details?id=com.kddi.android.UtaPass&hl=ja&showAllReviews=true) reviews on Google Play.
2. [KKBOX](https://play.google.com/store/apps/details?id=com.skysoft.kkbox.android&hl=ja&showAllReviews=true) reviews on Google Play.

## Bottleneck

- Is text pre-processing (e.g. remove stop words, remove punctuation, remove bad characters) neccessary?
- Tokenise in character-level or word-level?
- Do every reviews have sentiment words or charateristic of polarity?
- Does this dataset exist an imbalance problem?

## Flow Chart of Text Classification

![Flow Chart](https://github.com/penguinwang96825/Text-Classifier-for-UtaPass-and-KKBOX/raw/master/image/flowChart.jpg)

## Workstation

- Processor: Intel Core i9-9900K
- Motherboard: Gigabyte Z390 AORUS MASTER
- GPU: MSI RTX2080Ti Gaming X Trio 11G
- RAM: Kingston 64GB DDR4-3000 HyperX Predator
- CPU Cooler: MasterLiquid ML240L
- Storage: PLEXTOR M9PeGN 1TB M.2 2280 PCIe SSD
- Power: Antec HCG750 Gold
- Case: Fractal Design R6-BKO-TG

# Preparation

1. Preparing Selenium, BeautifulSoup, and Pandas.
 - Selenium: Selenium is an open source tool used for automating.
 - Beautiful Soup: BeautifulSoup is a Python library for parsing data out of HTML and XML files.
 - Pandas: Pandas is an open source data analysis tools for the Python programming language.

2. Install [MeCab](https://qiita.com/yukinoi/items/990b6933d9f21ba0fb43) on win10.
 - Download [MeCab](https://github.com/ikegami-yukino/mecab/releases/download/v0.996.2/mecab-64-0.996.2.exe) 64bit version first.
 - Run pip install `https://github.com/ikegami-yukino/mecab/archive/v0.996.2.tar.gz` in terminal.
 - Run `python -m pip install mecab` in terminal.

There are other tools which can deal with other languages.

* English Segmentation Tools
  * [NLTK](https://github.com/nltk/nltk)
  * [spaCy](https://spacy.io/api/tokenizer)
  * [SentencePiece](https://github.com/google/sentencepiece)
  * [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP)

* Chinese Segmentation Tools
  * [Jieba](https://github.com/fxsjy/jieba)
  * [SnowNLP](https://github.com/isnowfy/snownlp)
  * [LTP](https://www.ltp-cloud.com/)
  * [HanNLP](https://github.com/hankcs/HanLP)
  * [PKUSEG](https://github.com/lancopku/pkuseg-python)

* Japanese Segmentation Tools
  * [MeCab](https://github.com/ikegami-yukino/mecab/releases)
  * [Fugashi](https://www.dampfkraft.com/nlp/how-to-tokenize-japanese.html)
  * [Janome](https://mocobeta.github.io/janome/en/)

# Main Code for Crawling

Data Crawling is to deal with large data-sets where you develop your crawlers (or bots) which crawl to the deepest of the web pages. In this project, I am going to crawl all the reviews related to UtaPass from Google Play.

## Scroll-down Feature and Click-button Feature

In Google Play, you need to scroll down pages to reach all the contents, in this project, I utilise Selenium to do so.

```python
def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True

def scroll_ownPage():
    # Xpath of "???????????????" bottom
    button = '//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div[1]/div/div/div[1]/div[2]/div[2]/div/span/span'
    
    # Keep scrolling down until to the very bottom
    keep_scrolling = True
    while keep_scrolling:
        try: 
            # Scroll down to the bottom
            for _ in range(5):
                try: 
                    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                    time.sleep(1 + random.random())
                except:
                    break
            # Click "???????????????"
            if check_exists_by_xpath(button):
                driver.find_element_by_xpath(button).click()
                time.sleep(2 + random.random())
            else:
                # Stop scrolling down
                keep_scrolling = False
        except: 
            pass
```

## Start Crawling

```python
def open_google_play_reviews(url):
    driver_path = r"C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\chromedriver.exe"
    driver = webdriver.Chrome(driver_path)
    time.sleep(2 + random.random())
    driver.get(url)
    time.sleep(5 + random.random())
    
    scrollDownPage()
    soup = BeautifulSoup(driver.page_source, "html.parser")
    time.sleep(3 + random.random())
    driver.quit()
    return soup
    
def convert_soup_to_dataframe(soup):
    reviews = soup.find(name="div", attrs={"jsname": "fk8dgd"})
    reviews_list = reviews.find_all(name="div", attrs={"jscontroller": "H6eOGe"})
    reviews_all = []
    for i in range(len(reviews_list)):
        name = reviews_list[i].find(name="span", attrs={"class": "X43Kjb"}).string
        date = reviews_list[i].find(name="span", attrs={"class": "p2TkOb"}).string
        rating = reviews_list[i].find(name="div", attrs={"class": "pf5lIe"}).find(name="div").get("aria-label")
        rating = int(rating.split("/")[0][-1])
        content = reviews_list[i].find(name="span", attrs={"jsname": "bN97Pc"}).string
        like = reviews_list[i].find(name="div", attrs={"class": "jUL89d y92BAb"}).string
        reviews_all.append([name, date, rating, content, like])
    df = pd.DataFrame(reviews_all)
    df.columns = ["name", "date", "rating", "content", "like"]
    return df
    
def crawl(url):
    print("Parsing soup from url...")
    soup = open_google_play_reviews(url)
    print("Done parsing soup from url.")
    df = convert_soup_to_dataframe(soup)
    return df
```

## Data Storage

There are 12498 reviews in total. Take a look at the dataframe.

||Author Name|Review Date|Reviewer Ratings|Review Content|
|---|---|---|---|---|
|195|????????????|2018???12???4???|1|??????????????????????????????????????????????????????|
|13|?????????|2019???4???22???|1|LISMO???????????????????????????????????????LISMO??????|
|47|????????????|2019???3???12???|3|???????????????????????????????????????????????????????????????????????????????????????????????????????????????|
|142|????????????|2019???2???14???|4|?????????????????????|
|45|???????????????|2019???4???27???|1|???????????????????????????|

# Main Code for Modelling

## Import Packages

```python
import warnings
import re
import emoji
import MeCab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import ZeroPadding1D
from keras.layers import Dropout
from keras.layers import SpatialDropout1D
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPool1D
from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import SimpleRNN
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from gensim.models.word2vec import Word2Vec
```

## Load Data In

First, split dataframe into two categories: positive and negative. Second, do some text preprocessing. For instance, if rating is lower than 3 stars, label it as negative.

```python
df = pd.read_csv(r"C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\data\all_20200423.csv")
# create the label
df["label"] = df["rating"].apply(lambda x: 0 if int(x) <= 3 else 1)
# select only relevant columns
df = df[["content", "label"]]
df["content"] = df["content"].map(str)
df.head(5)
```

---

||content|label|
|---|---|---|
|0|??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????...|0|
|1|??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????...|0|
|2|???????????????LISMO??????????????????????????????LISMO????????????????????????????????????????????????LISMO...|0|
|3|?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????|0|
|4|LISMO???????????????????????????????????????LISMO????????????????????????????????????LISMO??????????????????...|0|

## Data Pre-processing

1. Remove emoji.
2. Remove punctuation
3. Remove digits.
4. Tokenise sentence using MeCab.

```python
def create_mecab_list(text):
    pos_list = [10, 11, 31, 32, 34]
    pos_list.extend(list(range(36,50)))
    pos_list.extend([59, 60, 62, 67])

    mecab_list = []
    mecab = MeCab.Tagger("-Owakati")
    mecab.parse("")
    # encoding = text.encode('utf-8')
    node = mecab.parseToNode(text)
    while node:
        if len(node.surface) > 1:
            if node.posid in pos_list:
                morpheme = node.surface
                mecab_list.append(morpheme)
        node = node.next
    return mecab_list

def give_emoji_free_text(text):
    allchars = [string for string in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    cleaned_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return cleaned_text

def clean_text(text):
    # Remove emoji
    text = give_emoji_free_text(text)
    # Remove punctuation
    text = re.sub(r'[^\w\d\s]+', '', text)
    # Remove digits
    text = ''.join([i for i in text if not i.isdigit()]) 
    # Tokenize the sentence
    tokenised_text_list = create_mecab_list(text)
    return tokenised_text_list
```

## Main Code for Modelling

### Set Parameters

```python
# Input parameters
config = {
    # Text parameters
    "MAX_FEATURE": 10000, 
    "MAX_LEN": 64, 
    "EMBED_SIZE": 300, 

    # Convolution parameters
    "filter_length": 3, 
    "nb_filter": 150, 
    "pool_length": 2, 
    "cnn_activation": 'relu', 
    "border_mode": 'same', 

    # RNN parameters
    "lstm_cell": 128, 
    "output_size": 50, 
    "rnn_activation": 'tanh', 
    "recurrent_activation": 'hard_sigmoid', 
    
    # FC and Dropout
    "fc_cell": 128, 
    "dropout_rate": 0.25, 

    # Compile parameters
    "loss": 'binary_crossentropy', 
    "optimizer": 'adam', 

    # Training parameters
    "batch_size": 256, 
    "nb_epoch": 1000, 
    "validation_split": 0.20, 
    "shuffle": True
}
```

### Create Word Index

```python
def create_word2index_and_index2word(df):
    df["cleaned_text"] = df["content"].apply(clean_text)
    sum_list = []
    for index, row in df.iterrows():
        sum_list += row["cleaned_text"]

    word2index = dict()
    index2word = dict()
    num_words = 0
    for word in sum_list:
        if word not in word2index:
            # First entry of word into vocabulary
            word2index[word] = num_words
            index2word[num_words] = word
            num_words += 1
    
    return word2index, index2word

def convert_tokens_to_ids(tokens_list, word2index):
    ids_list = []
    for token in tokens_list:
        if word2index.get(token, None) != None:
            ids_list.append(word2index[token])
    return ids_list

def remove_empty_ids_rows(df):
    empty = (df['ids'].map(len) == 0)
    return df[~empty]

word2index, index2word = create_word2index_and_index2word(df)
df["ids"] = df["content"].apply(lambda x: convert_tokens_to_ids(clean_text(x), word2index))
df = remove_empty_ids_rows(df)
df.head()
```

### Split Data

Split the data into training data (80%) and testing data (20%).
* Training set: a subset to train a model
* Testing set: a subset to test the trained model

```python
x = df["ids"].map(lambda x: np.array(x))
x = sequence.pad_sequences(x, maxlen=config["MAX_LEN"], padding="post")
print("Features: \n", x)
y = df["label"].values
print("Labels: \n", y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)
```

### Word Embedding

Build the word2vec model to do word embedding.

Reference:
* https://github.com/philipperemy/japanese-words-to-vectors/blob/master/README.md)
* http://jalammar.github.io/illustrated-word2vec/

Training a Japanese Wikipedia Word2Vec Model by Gensim and Mecab:
* Kyubyong Park's [GitHub](https://github.com/Kyubyong/wordvectors)
* Omuram's [Qiita](https://qiita.com/omuram/items/6570973c090c6f0cb060)
* TextMiner's [Website](https://textminingonline.com/training-a-japanese-wikipedia-word2vec-model-by-gensim-and-mecab)

```python
def get_embedding_index(model_path):
    w2v = Word2Vec.load(model_path)
    embedding_index = {}
    for word in w2v.wv.vocab:
        embedding_index[word] = w2v.wv.word_vec(word)
    print('Loaded {} word vectors.'.format(len(embedding_index)))

    return embedding_index

def get_embedding_matrix(word2index, embeddings_index, embed_size):
    embedding_matrix = np.zeros((len(word2index) + 1, embed_size))
    for word, i in word2index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words found in embedding index will be pretrained vectors.
            embedding_matrix[i+1] = embedding_vector
        else:
            # words not found in embedding index will be random vectors with certain mean&std.
            embedding_matrix[i+1] = np.random.normal(0.053, 0.3146, size=(1, embed_size))[0]

    # save embedding matrix
    # embed_df = pd.DataFrame(embedding_matrix)
    # embed_df.to_csv(self.path_embedding_matrix, header=None, sep=' ')

    return embedding_matrix

embedding_index = get_embedding_index(
    r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\word2vec\ja.bin')
embedding_matrix = get_embedding_matrix(word2index, embedding_index, embed_size=300)
```

### Build Model

Construct neural network architectures.

Reference:
* Asanilta Fahda's [GitHub](https://github.com/asanilta/amazon-sentiment-keras-experiment)
* teratsyk's [GitHub](https://github.com/teratsyk/bokete-ai)

Build a customised metrics function to record f1 value after each epoch.

```python
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
```

When training a neural network, f1 score is an important metric to evaluate the performance of classification models, especially for unbalanced classes where the binary accuracy is useless. So I biuld two helper functions `get_f1` and `predict`.

```python
def predict(sentences, model):
    y_prob = model.predict(x_test)
    y_prob = y_prob.squeeze()
    y_pred = (y_prob > 0.5) 
    return y_pred
```

Set optimisers to update gradient.

```python
adagrad = Adagrad(learning_rate=0.01)
adadelta = Adadelta(learning_rate=0.01, rho=0.95)
rmsprop = RMSprop(learning_rate=0.001, rho=0.9)
nadam = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
```

#### Simple RNN
```python
def train_simple_rnn(x_train, y_train, wv_matrix, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = SimpleRNN(
        units=config["output_size"], 
        activation=config["rnn_activation"], 
        kernel_regularizer=regularizers.l2(0.3), 
        return_sequences=True)(x)
    x = SimpleRNN(
        units=config["output_size"], 
        activation=config["rnn_activation"], 
        kernel_regularizer=regularizers.l2(0.3), 
        return_sequences=False)(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    if not load_weights: 
        model.compile(loss=config["loss"], 
                      optimizer=adagrad, 
                      metrics=[get_f1])

        print("="*20, "Start Training Simple RNN", "="*20)

        path = r'C:\Users\YangWang\Desktop\UtaPass_KKBOX_Classifier\weights\rnn_weights.hdf5'
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        history = model.fit(
            x_train, y_train, 
            batch_size=config['batch_size'], 
            epochs=config['nb_epoch'], 
            validation_split=config['validation_split'], 
            shuffle=config['shuffle'], 
            verbose=0, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights:
        history = None
        path = r'C:\Users\YangWang\Desktop\UtaPass_KKBOX_Classifier\weights\rnn_weights.hdf5'
        model.load_weights(path)
    
    return history, model
```

#### GRU
```python
def train_gru(x_train, y_train, wv_matrix, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = GRU(units=config["output_size"], return_sequences=True, dropout=config['dropout_rate'])(x)
    x = GRU(units=config["output_size"], return_sequences=False, dropout=config['dropout_rate'])(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    if not load_weights:
        model.compile(loss=config["loss"], 
                      optimizer=rmsprop, 
                      metrics=[get_f1])

        print("="*20, "Start Training GRU", "="*20)

        path = r'C:\Users\YangWang\Desktop\UtaPass_KKBOX_Classifier\weights\gru_weights.hdf5'
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        history = model.fit(
            x_train, y_train, 
            batch_size=config['batch_size'], 
            epochs=config['nb_epoch'], 
            validation_split=config['validation_split'], 
            shuffle=config['shuffle'], 
            verbose=verbose, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights:
        history = None
        path = r'C:\Users\YangWang\Desktop\UtaPass_KKBOX_Classifier\weights\gru_weights.hdf5'
        model.load_weights(path)
    
    return history, model
```

#### LSTM
```python
def train_lstm(x_train, y_train, wv_matrix, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = LSTM(units=config['lstm_cell'], return_sequences=True, kernel_regularizer=regularizers.l2(0.3))(x)
    x = LSTM(units=config['lstm_cell'], return_sequences=False, kernel_regularizer=regularizers.l2(0.3))(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    if not load_weights: 
        model.compile(loss=config["loss"], 
                      optimizer=adagrad, 
                      metrics=[get_f1])

        print("="*20, "Start Training LSTM", "="*20)

        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\lstm_weights.hdf5'
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        history = model.fit(
            x_train, y_train, 
            batch_size=config['batch_size'], 
            epochs=config['nb_epoch'], 
            validation_split=config['validation_split'], 
            shuffle=config['shuffle'], 
            verbose=verbose, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights:
        history = None
        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\lstm_weights.hdf5'
        model.load_weights(path)
    
    return history, model
```

#### BiLSTM
```python
def train_bilstm(x_train, y_train, wv_matrix, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = Bidirectional(LSTM(
        units=config['lstm_cell'], 
        return_sequences=True, 
        dropout=config['dropout_rate'], 
        kernel_regularizer=regularizers.l2(0.3)))(x)
    x = Bidirectional(LSTM(
        units=config['lstm_cell'], 
        return_sequences=False, 
        kernel_regularizer=regularizers.l2(0.3)))(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(config['fc_cell'])(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    if not load_weights:
        model.compile(loss=config["loss"], 
                      optimizer=adagrad, 
                      metrics=[get_f1])

        print("="*20, "Start Training BiLSTM", "="*20)

        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\bilstm_weights.hdf5'
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        history = model.fit(
            x_train, y_train, 
            batch_size=config['batch_size'], 
            epochs=config['nb_epoch'], 
            validation_split=config['validation_split'], 
            shuffle=config['shuffle'], 
            verbose=verbose, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights:
        history = None
        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\bilstm_weights.hdf5'
        model.load_weights(path)
    
    return history, model
```

#### Attention
```python
def train_attention(x_train, y_train, wv_matrix, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = SeqSelfAttention(units=128, kernel_regularizer=regularizers.l2(0.1))(x)
    x = SeqSelfAttention(units=128, kernel_regularizer=regularizers.l2(0.1))(x)
    x = SeqSelfAttention(units=128, kernel_regularizer=regularizers.l2(0.1))(x)
    x = SeqSelfAttention(units=128, kernel_regularizer=regularizers.l2(0.1))(x)
    x = SeqSelfAttention(units=128, kernel_regularizer=regularizers.l2(0.1))(x)
    x = SeqSelfAttention(units=128, kernel_regularizer=regularizers.l2(0.1))(x)
    x = SeqSelfAttention(units=128, kernel_regularizer=regularizers.l2(0.1))(x)
    x = SeqSelfAttention(units=128, kernel_regularizer=regularizers.l2(0.1))(x)
    x = Flatten()(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    if not load_weights:
        model.compile(loss=config["loss"], 
                      optimizer=adagrad, 
                      metrics=[get_f1])

        print("="*20, "Start Training Attention", "="*20)

        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\attention_weights.hdf5'
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001, verbose=1)

        history = model.fit(
            x_train, y_train, 
            batch_size=config['batch_size'], 
            epochs=config['nb_epoch'], 
            validation_split=config['validation_split'], 
            shuffle=config['shuffle'], 
            verbose=verbose, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights:
        history = None
        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\attention_weights.hdf5'
        model.load_weights(path)
    
    return history, model
```

#### CNN-LSTM
```python
def train_cnn_lstm(x_train, y_train, wv_matrix, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    
    x = Conv1D(filters=config['nb_filter'], kernel_size=config['filter_length'], padding=config['border_mode'])(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=config['pool_length'])(x)
    
    x = Conv1D(filters=config['nb_filter']*2, kernel_size=config['filter_length'], padding=config['border_mode'])(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=config['pool_length'])(x)
    
    x = Conv1D(filters=config['nb_filter']*4, kernel_size=config['filter_length'], padding=config['border_mode'])(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=config['pool_length'])(x)
    
    x = LSTM(units=config['lstm_cell'], return_sequences=True, kernel_regularizer=regularizers.l2(0.3))(x)
    x = SeqSelfAttention(units=config['lstm_cell'], kernel_regularizer=regularizers.l2(0.1))(x)
    
    x = Flatten()(x)
    x = Dense(config['fc_cell'])(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    if not load_weights:
        model.compile(loss=config['loss'], 
                      optimizer=adadelta, 
                      metrics=[get_f1])

        print("="*20, "Start Training CNN-LSTM", "="*20)

        path = r'C:\Users\YangWang\Desktop\UtaPass_KKBOX_Classifier\weights\{}_weights.hdf5'.format("cnn_lstm")
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        history = model.fit(
            x_train, y_train, 
            batch_size=config['batch_size'], 
            epochs=config['nb_epoch'], 
            validation_split=config['validation_split'], 
            shuffle=config['shuffle'], 
            verbose=verbose, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights: 
        history = None
        path = r'C:\Users\YangWang\Desktop\UtaPass_KKBOX_Classifier\weights\{}_weights.hdf5'.format("cnn_lstm")
        model.load_weights(path)
    
    return history, model
```

#### CNN-static
Based on "Convolutional Neural Networks for Sentence Classification" written by Yoon Kim [[paper link](http://arxiv.org/pdf/1408.5882v2.pdf)]
```python
# Yoon Kim's paper: Convolutional Neural Networks for Sentence Classification
# Reference from https://www.aclweb.org/anthology/D14-1181.pdf
def train_cnn_static(x_train, y_train, wv_matrix, trainable=False, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=trainable)(x_input)
    
    x_conv_1 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=3, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(0.3))(x)
    x_conv_1 = BatchNormalization()(x_conv_1)
    x_conv_1 = Activation("relu")(x_conv_1)
    x_conv_1 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_1)
    
    x_conv_2 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=4, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(0.3))(x)
    x_conv_2 = BatchNormalization()(x_conv_2)
    x_conv_2 = Activation("relu")(x_conv_2)
    x_conv_2 = MaxPooling1D(pool_size=(config["MAX_LEN"]-4+1), strides=1, padding="valid")(x_conv_2)
    
    x_conv_3 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=5, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(0.3))(x)
    x_conv_3 = BatchNormalization()(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = MaxPooling1D(pool_size=(config["MAX_LEN"]-5+1), strides=1, padding="valid")(x_conv_3)
    
    main = Concatenate(axis=1)([x_conv_1, x_conv_2, x_conv_3])
    main = Flatten()(main)
    main = Dropout(config['dropout_rate'])(main)
    main = Dense(units=1)(main)
    main = Activation('sigmoid')(main)
    model = Model(inputs=x_input, outputs=main)

    if not load_weights:
        model.compile(loss=config["loss"], 
                      optimizer=adagrad, 
                      metrics=[get_f1])

        print("="*20, "Start Training CNN-static", "="*20)

        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\cnn_static_weights.hdf5'
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        history = model.fit(
            x_train, y_train, 
            batch_size=config['batch_size'], 
            epochs=config['nb_epoch'], 
            validation_split=config['validation_split'], 
            shuffle=config['shuffle'], 
            verbose=verbose, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights:
        history = None
        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\cnn_static_weights.hdf5'
        model.load_weights(path)
    
    return history, model
```

#### CNN-multichannel
Based on "Convolutional Neural Networks for Sentence Classification" written by Yoon Kim [[paper link](http://arxiv.org/pdf/1408.5882v2.pdf)]
```python
def train_cnn_multichannel(x_train, y_train, wv_matrix, trainable=False, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    
    # Channel 1
    x_input_1 = Input(shape=(config["MAX_LEN"], ))
    embedding_1 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_1)
    x_conv_1 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(0.3))(embedding_1)
    x_conv_1 = BatchNormalization()(x_conv_1)
    x_conv_1 = Activation("relu")(x_conv_1)
    x_conv_1 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_1)
    flat_1 = Flatten()(x_conv_1)
    
    # Channel 2
    x_input_2 = Input(shape=(config["MAX_LEN"], ))
    embedding_2 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_2)
    x_conv_2 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(0.3))(embedding_2)
    x_conv_2 = BatchNormalization()(x_conv_2)
    x_conv_2 = Activation("relu")(x_conv_2)
    x_conv_2 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_2)
    flat_2 = Flatten()(x_conv_2)
    
    # Channel 1
    x_input_3 = Input(shape=(config["MAX_LEN"], ))
    embedding_3 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_3)
    x_conv_3 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(0.3))(embedding_3)
    x_conv_3 = BatchNormalization()(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_3)
    flat_3 = Flatten()(x_conv_3)
    
    main = Concatenate(axis=1)([flat_1, flat_2, flat_3])
    main = Dense(units=100)(main)
    main = Activation('relu')(main)
    main = Dropout(config['dropout_rate'])(main)
    main = Dense(units=1)(main)
    main = Activation('sigmoid')(main)
    model = Model(inputs=[x_input_1, x_input_2, x_input_3], outputs=main)

    if not load_weights:
        model.compile(loss=config["loss"], 
                      optimizer=adagrad, 
                      metrics=[get_f1])

        print("="*20, "Start Training CNN-multichannel", "="*20)

        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\{}_weights.hdf5'.format("cnn_multi")
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        history = model.fit(
            [x_train, x_train, x_train], y_train, 
            batch_size=config['batch_size'], 
            epochs=config['nb_epoch'], 
            validation_split=config['validation_split'], 
            shuffle=config['shuffle'], 
            verbose=verbose, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights:
        history = None
        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\{}_weights.hdf5'.format("cnn_multi")
        model.load_weights(path)
    
    return history, model
```

#### Text-ResNet
```python
def identity_resnet_block(x, nb_filter):
    x_shortcut = x

    # First component of main path
    res_x = Conv1D(
        filters=nb_filter, 
        kernel_size=5, 
        strides=1, 
        padding='same')(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Second component of main path
    res_x = Conv1D(
        filters=nb_filter, 
        kernel_size=5, 
        strides=1, 
        padding='same')(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Third component of main path
    res_x = Conv1D(
        filters=nb_filter, 
        kernel_size=5, 
        strides=1, 
        padding='same')(res_x)
    res_x = BatchNormalization()(res_x)

    # Final Step: add shortcut value to the main path
    x = Add()([x_shortcut, res_x])
    output = Activation('relu')(x)
    return output

def convolutional_resnet_block(x, nb_filter):
    x_shortcut = x

    # First component of main path
    res_x = Conv1D(
        filters=nb_filter, 
        kernel_size=5, 
        strides=1, 
        padding='same')(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Second component of main path
    res_x = Conv1D(
        filters=nb_filter, 
        kernel_size=5, 
        strides=1, 
        padding='same')(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Third component of main path
    res_x = Conv1D(
        filters=nb_filter, 
        kernel_size=5, 
        strides=1, 
        padding='same')(res_x)
    res_x = BatchNormalization()(res_x)

    # Shortcut path
    x_shortcut = Conv1D(
        filters=nb_filter, 
        kernel_size=5, 
        strides=1, 
        padding='same')(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)

    # Final Step: add shortcut value to the main path
    x = Add()([x_shortcut, res_x])
    output = Activation('relu')(x)
    return output

def train_text_resnet(x_train, y_train, wv_matrix, verbose=0, load_weights=False):
    tf.keras.backend.clear_session()
    # Resnet for reviews of UtaPass and KKBOX
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(0.25)(x)

    # Stage 1
    x = Conv1D(filters=64, kernel_size=3, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    # Stage 2
    x = convolutional_resnet_block(x, 64)
    x = identity_resnet_block(x, 64)
    x = identity_resnet_block(x, 64)
    # Stage 3
    x = convolutional_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    # Stage 4
    x = convolutional_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    # Stage 5
    x = convolutional_resnet_block(x, 512)
    x = identity_resnet_block(x, 512)
    x = identity_resnet_block(x, 512)
    # Average pool
    x = AveragePooling1D(pool_size = 1)(x)
    # Output layer
    x = Flatten()(x)

    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dense(units=1)(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=x_input, outputs=x)
    
    if not load_weights: 
        model.compile(loss=config["loss"],
                      optimizer=Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                      metrics=[get_f1])

        print("="*20, "Start Training Text-ResNet", "="*20)

        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\text_resnet_weights.hdf5'
        model_checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        history = model.fit(
            x_train, y_train, 
            batch_size=config["batch_size"], 
            epochs=config["nb_epoch"], 
            validation_split=config["validation_split"], 
            shuffle=config["shuffle"], 
            verbose=verbose, 
            callbacks=[early_stopping, reduce_lr, model_checkpoint])
    if load_weights:
        history = None
        path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\weights\text_resnet_weights.hdf5'
        model.load_weights(path)
    
    return history, model
```

### Start Training

1. Train each model and get history and model architecure.
2. Load the checkpoint from weights folder we saved during training.
3. Predict sentiment probability from testing set.
4. Compute accuracy score, f1 score, and confusion matrix.
5. Plot train and val history (loss and f1) and visualise confusion matrix.

#### Visualisation
Define two ploting functions to visualise the history of accuracy and loss.

```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history_ggplot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
def plot_history(history):
    # plot results
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    f1 = history.history['acc']
    val_f1 = history.history['val_acc']

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.title('Loss')
    epochs = len(loss)
    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.set_facecolor('snow')
    plt.grid(color='lightgray', linestyle='-', linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(range(epochs), f1, marker='.', label='acc')
    plt.plot(range(epochs), val_f1, marker='.', label='val_acc')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.set_facecolor('snow')
    plt.grid(color='lightgray', linestyle='-', linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    
    plt.tight_layout()
    plt.show()
```

Plot confusion matrix heatmap.

```python
def plot_confusion_matrix(cm):
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize=(10, 8))
    hmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha="right")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Target Sentiment")
    plt.show()
```

### Accuracy, F1 Score, and Confusion Matrix
Compare the performance among several deep learning models.

#### Simple RNN

Accuracy:  0.7419 
F1 Score:  0.7773

<details>
<summary>Simple RNN Model Architecture</summary>
<pre><code>
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 64)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 64, 300)           2251500   
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 64, 300)           0         
_________________________________________________________________
simple_rnn_1 (SimpleRNN)     (None, 64, 50)            17550     
_________________________________________________________________
simple_rnn_2 (SimpleRNN)     (None, 50)                5050      
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51        
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0         
=================================================================
Total params: 2,274,151
Trainable params: 22,651
Non-trainable params: 2,251,500
_________________________________________________________________
</code></pre>
</details>

#### GRU

Accuracy:  0.7821
F1 Score:  0.8216

<details>
<summary>GRU Model Architecture</summary>
<pre><code>
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 64)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 64, 300)           2251500   
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 64, 300)           0         
_________________________________________________________________
gru_1 (GRU)                  (None, 64, 50)            52650     
_________________________________________________________________
gru_2 (GRU)                  (None, 50)                15150     
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51        
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0         
=================================================================
Total params: 2,319,351
Trainable params: 67,851
Non-trainable params: 2,251,500
_________________________________________________________________
</code></pre>
</details>

#### LSTM

Accuracy:  0.7697
F1 Score:  0.7945

<details>
<summary>LSTM Model Architecture</summary>
<pre><code>
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 64)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 64, 300)           2251500   
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 64, 300)           0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 64, 64)            93440     
_________________________________________________________________
lstm_2 (LSTM)                (None, 64)                33024     
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65        
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0         
=================================================================
Total params: 2,378,029
Trainable params: 126,529
Non-trainable params: 2,251,500
_________________________________________________________________
</code></pre>
</details>

#### BiLSTM

Accuracy:  0.7430
F1 Score:  0.7696

<details>
<summary>BiLSTM Model Architecture</summary>
<pre><code>
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 64)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 64, 300)           2251500   
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 64, 300)           0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 64, 128)           186880    
_________________________________________________________________
bidirectional_2 (Bidirection (None, 128)               98816     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0         
=================================================================
Total params: 2,553,837
Trainable params: 302,337
Non-trainable params: 2,251,500
_________________________________________________________________
</code></pre>
</details>

#### Attention

Accuracy:  0.7496
F1 Score:  0.7857

<details>
<summary>Attention Model Architecture</summary>
<pre><code>
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 64)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 64, 300)           2251500   
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 64, 300)           0         
_________________________________________________________________
seq_self_attention_1 (SeqSel (None, 64, 300)           77057     
_________________________________________________________________
seq_self_attention_2 (SeqSel (None, 64, 300)           77057     
_________________________________________________________________
seq_self_attention_3 (SeqSel (None, 64, 300)           77057     
_________________________________________________________________
flatten_1 (Flatten)          (None, 19200)             0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 19200)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 19201     
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0         
=================================================================
Total params: 2,501,872
Trainable params: 250,372
Non-trainable params: 2,251,500
_________________________________________________________________
</code></pre>
</details>

#### CNN-Static

Accuracy:  0.7736
F1 Score:  0.8031

<details>
<summary>CNN-Static Model Architecture</summary>
<pre><code>
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 64)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 64, 300)      2251500     input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 64, 150)      135150      embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 64, 150)      180150      embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 64, 150)      225150      embedding_1[0][0]                
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 64, 150)      600         conv1d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 64, 150)      600         conv1d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 64, 150)      600         conv1d_3[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 64, 150)      0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 150)      0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 64, 150)      0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 3, 150)       0           activation_1[0][0]               
__________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)  (None, 4, 150)       0           activation_2[0][0]               
__________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)  (None, 5, 150)       0           activation_3[0][0]               
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 12, 150)      0           max_pooling1d_1[0][0]            
                                                                 max_pooling1d_2[0][0]            
                                                                 max_pooling1d_3[0][0]            
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 1800)         0           concatenate_1[0][0]              
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 1800)         0           flatten_1[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            1801        dropout_1[0][0]                  
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 1)            0           dense_1[0][0]                    
==================================================================================================
Total params: 2,795,551
Trainable params: 543,151
Non-trainable params: 2,252,400
__________________________________________________________________________________________________
</code></pre>
</details>

#### CNN-MultiChannel

Accuracy:  0.7744
F1 Score:  0.8073

<details>
<summary>CNN-MultiChannel Model Architecture</summary>
<pre><code>
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 64)           0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 64)           0                                            
__________________________________________________________________________________________________
input_3 (InputLayer)            (None, 64)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 64, 300)      2251500     input_1[0][0]                    
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 64, 300)      2251500     input_2[0][0]                    
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 64, 300)      2251500     input_3[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 64, 150)      135150      embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 64, 150)      135150      embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 64, 150)      135150      embedding_3[0][0]                
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 64, 150)      600         conv1d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 64, 150)      600         conv1d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 64, 150)      600         conv1d_3[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 64, 150)      0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 150)      0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 64, 150)      0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 3, 150)       0           activation_1[0][0]               
__________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)  (None, 3, 150)       0           activation_2[0][0]               
__________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)  (None, 3, 150)       0           activation_3[0][0]               
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 450)          0           max_pooling1d_1[0][0]            
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 450)          0           max_pooling1d_2[0][0]            
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 450)          0           max_pooling1d_3[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1350)         0           flatten_1[0][0]                  
                                                                 flatten_2[0][0]                  
                                                                 flatten_3[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 100)          135100      concatenate_1[0][0]              
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 100)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 100)          0           activation_4[0][0]               
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            101         dropout_1[0][0]                  
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 1)            0           dense_2[0][0]                    
==================================================================================================
Total params: 7,296,951
Trainable params: 541,551
Non-trainable params: 6,755,400
__________________________________________________________________________________________________
</code></pre>
</details>

#### CNN-LSTM

Accuracy:  0.7380
F1 Score:  0.7785

<details>
<summary>CNN-LSTM Model Architecture</summary>
<pre><code>
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 64)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 64, 300)           2251500   
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 64, 300)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 64, 150)           135150    
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 150)           600       
_________________________________________________________________
activation_1 (Activation)    (None, 64, 150)           0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 32, 150)           0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 32, 300)           135300    
_________________________________________________________________
batch_normalization_2 (Batch (None, 32, 300)           1200      
_________________________________________________________________
activation_2 (Activation)    (None, 32, 300)           0         
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 16, 300)           0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 16, 600)           540600    
_________________________________________________________________
batch_normalization_3 (Batch (None, 16, 600)           2400      
_________________________________________________________________
activation_3 (Activation)    (None, 16, 600)           0         
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 8, 600)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 8, 64)             170240    
_________________________________________________________________
seq_self_attention_1 (SeqSel (None, 8, 64)             8321      
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               65664     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
_________________________________________________________________
activation_4 (Activation)    (None, 1)                 0         
=================================================================
Total params: 3,311,104
Trainable params: 1,057,504
Non-trainable params: 2,253,600
_________________________________________________________________
</code></pre>
</details>

#### Text-ResNet

Accuracy:  0.7283
F1 Score:  0.7535

<details>
<summary>Text-ResNet Model Architecture</summary>
<pre><code>
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 64)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 64, 300)      2251500     input_1[0][0]                    
__________________________________________________________________________________________________
spatial_dropout1d_1 (SpatialDro (None, 64, 300)      0           embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 31, 64)       57664       spatial_dropout1d_1[0][0]        
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 31, 64)       256         conv1d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 31, 64)       0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 15, 64)       0           activation_1[0][0]               
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 15, 64)       20544       max_pooling1d_1[0][0]            
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 15, 64)       256         conv1d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 15, 64)       0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 15, 64)       20544       activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 15, 64)       256         conv1d_3[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 15, 64)       0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 15, 64)       20544       max_pooling1d_1[0][0]            
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 15, 64)       20544       activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 15, 64)       256         conv1d_5[0][0]                   
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 15, 64)       256         conv1d_4[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 15, 64)       0           batch_normalization_5[0][0]      
                                                                 batch_normalization_4[0][0]      
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 15, 64)       0           add_1[0][0]                      
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 15, 64)       20544       activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 15, 64)       256         conv1d_6[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 15, 64)       0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 15, 64)       20544       activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 15, 64)       256         conv1d_7[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 15, 64)       0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 15, 64)       20544       activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 15, 64)       256         conv1d_8[0][0]                   
__________________________________________________________________________________________________
add_2 (Add)                     (None, 15, 64)       0           activation_4[0][0]               
                                                                 batch_normalization_8[0][0]      
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 15, 64)       0           add_2[0][0]                      
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 15, 64)       20544       activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 15, 64)       256         conv1d_9[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 15, 64)       0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv1d_10 (Conv1D)              (None, 15, 64)       20544       activation_8[0][0]               
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 15, 64)       256         conv1d_10[0][0]                  
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 15, 64)       0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv1d_11 (Conv1D)              (None, 15, 64)       20544       activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 15, 64)       256         conv1d_11[0][0]                  
__________________________________________________________________________________________________
add_3 (Add)                     (None, 15, 64)       0           activation_7[0][0]               
                                                                 batch_normalization_11[0][0]     
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 15, 64)       0           add_3[0][0]                      
__________________________________________________________________________________________________
conv1d_12 (Conv1D)              (None, 15, 128)      41088       activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 15, 128)      512         conv1d_12[0][0]                  
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 15, 128)      0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv1d_13 (Conv1D)              (None, 15, 128)      82048       activation_11[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 15, 128)      512         conv1d_13[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 15, 128)      0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
conv1d_15 (Conv1D)              (None, 15, 128)      41088       activation_10[0][0]              
__________________________________________________________________________________________________
conv1d_14 (Conv1D)              (None, 15, 128)      82048       activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 15, 128)      512         conv1d_15[0][0]                  
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 15, 128)      512         conv1d_14[0][0]                  
__________________________________________________________________________________________________
add_4 (Add)                     (None, 15, 128)      0           batch_normalization_15[0][0]     
                                                                 batch_normalization_14[0][0]     
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 15, 128)      0           add_4[0][0]                      
__________________________________________________________________________________________________
conv1d_16 (Conv1D)              (None, 15, 128)      82048       activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 15, 128)      512         conv1d_16[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 15, 128)      0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv1d_17 (Conv1D)              (None, 15, 128)      82048       activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 15, 128)      512         conv1d_17[0][0]                  
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 15, 128)      0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
conv1d_18 (Conv1D)              (None, 15, 128)      82048       activation_15[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 15, 128)      512         conv1d_18[0][0]                  
__________________________________________________________________________________________________
add_5 (Add)                     (None, 15, 128)      0           activation_13[0][0]              
                                                                 batch_normalization_18[0][0]     
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 15, 128)      0           add_5[0][0]                      
__________________________________________________________________________________________________
conv1d_19 (Conv1D)              (None, 15, 128)      82048       activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 15, 128)      512         conv1d_19[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 15, 128)      0           batch_normalization_19[0][0]     
__________________________________________________________________________________________________
conv1d_20 (Conv1D)              (None, 15, 128)      82048       activation_17[0][0]              
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 15, 128)      512         conv1d_20[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 15, 128)      0           batch_normalization_20[0][0]     
__________________________________________________________________________________________________
conv1d_21 (Conv1D)              (None, 15, 128)      82048       activation_18[0][0]              
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 15, 128)      512         conv1d_21[0][0]                  
__________________________________________________________________________________________________
add_6 (Add)                     (None, 15, 128)      0           activation_16[0][0]              
                                                                 batch_normalization_21[0][0]     
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 15, 128)      0           add_6[0][0]                      
__________________________________________________________________________________________________
conv1d_22 (Conv1D)              (None, 15, 128)      82048       activation_19[0][0]              
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 15, 128)      512         conv1d_22[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 15, 128)      0           batch_normalization_22[0][0]     
__________________________________________________________________________________________________
conv1d_23 (Conv1D)              (None, 15, 128)      82048       activation_20[0][0]              
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 15, 128)      512         conv1d_23[0][0]                  
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 15, 128)      0           batch_normalization_23[0][0]     
__________________________________________________________________________________________________
conv1d_24 (Conv1D)              (None, 15, 128)      82048       activation_21[0][0]              
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 15, 128)      512         conv1d_24[0][0]                  
__________________________________________________________________________________________________
add_7 (Add)                     (None, 15, 128)      0           activation_19[0][0]              
                                                                 batch_normalization_24[0][0]     
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 15, 128)      0           add_7[0][0]                      
__________________________________________________________________________________________________
conv1d_25 (Conv1D)              (None, 15, 256)      164096      activation_22[0][0]              
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 15, 256)      1024        conv1d_25[0][0]                  
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 15, 256)      0           batch_normalization_25[0][0]     
__________________________________________________________________________________________________
conv1d_26 (Conv1D)              (None, 15, 256)      327936      activation_23[0][0]              
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 15, 256)      1024        conv1d_26[0][0]                  
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 15, 256)      0           batch_normalization_26[0][0]     
__________________________________________________________________________________________________
conv1d_28 (Conv1D)              (None, 15, 256)      164096      activation_22[0][0]              
__________________________________________________________________________________________________
conv1d_27 (Conv1D)              (None, 15, 256)      327936      activation_24[0][0]              
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 15, 256)      1024        conv1d_28[0][0]                  
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 15, 256)      1024        conv1d_27[0][0]                  
__________________________________________________________________________________________________
add_8 (Add)                     (None, 15, 256)      0           batch_normalization_28[0][0]     
                                                                 batch_normalization_27[0][0]     
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 15, 256)      0           add_8[0][0]                      
__________________________________________________________________________________________________
conv1d_29 (Conv1D)              (None, 15, 256)      327936      activation_25[0][0]              
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 15, 256)      1024        conv1d_29[0][0]                  
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 15, 256)      0           batch_normalization_29[0][0]     
__________________________________________________________________________________________________
conv1d_30 (Conv1D)              (None, 15, 256)      327936      activation_26[0][0]              
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 15, 256)      1024        conv1d_30[0][0]                  
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 15, 256)      0           batch_normalization_30[0][0]     
__________________________________________________________________________________________________
conv1d_31 (Conv1D)              (None, 15, 256)      327936      activation_27[0][0]              
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 15, 256)      1024        conv1d_31[0][0]                  
__________________________________________________________________________________________________
add_9 (Add)                     (None, 15, 256)      0           activation_25[0][0]              
                                                                 batch_normalization_31[0][0]     
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 15, 256)      0           add_9[0][0]                      
__________________________________________________________________________________________________
conv1d_32 (Conv1D)              (None, 15, 256)      327936      activation_28[0][0]              
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 15, 256)      1024        conv1d_32[0][0]                  
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 15, 256)      0           batch_normalization_32[0][0]     
__________________________________________________________________________________________________
conv1d_33 (Conv1D)              (None, 15, 256)      327936      activation_29[0][0]              
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 15, 256)      1024        conv1d_33[0][0]                  
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 15, 256)      0           batch_normalization_33[0][0]     
__________________________________________________________________________________________________
conv1d_34 (Conv1D)              (None, 15, 256)      327936      activation_30[0][0]              
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 15, 256)      1024        conv1d_34[0][0]                  
__________________________________________________________________________________________________
add_10 (Add)                    (None, 15, 256)      0           activation_28[0][0]              
                                                                 batch_normalization_34[0][0]     
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 15, 256)      0           add_10[0][0]                     
__________________________________________________________________________________________________
conv1d_35 (Conv1D)              (None, 15, 256)      327936      activation_31[0][0]              
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 15, 256)      1024        conv1d_35[0][0]                  
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 15, 256)      0           batch_normalization_35[0][0]     
__________________________________________________________________________________________________
conv1d_36 (Conv1D)              (None, 15, 256)      327936      activation_32[0][0]              
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 15, 256)      1024        conv1d_36[0][0]                  
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 15, 256)      0           batch_normalization_36[0][0]     
__________________________________________________________________________________________________
conv1d_37 (Conv1D)              (None, 15, 256)      327936      activation_33[0][0]              
__________________________________________________________________________________________________
batch_normalization_37 (BatchNo (None, 15, 256)      1024        conv1d_37[0][0]                  
__________________________________________________________________________________________________
add_11 (Add)                    (None, 15, 256)      0           activation_31[0][0]              
                                                                 batch_normalization_37[0][0]     
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 15, 256)      0           add_11[0][0]                     
__________________________________________________________________________________________________
conv1d_38 (Conv1D)              (None, 15, 256)      327936      activation_34[0][0]              
__________________________________________________________________________________________________
batch_normalization_38 (BatchNo (None, 15, 256)      1024        conv1d_38[0][0]                  
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 15, 256)      0           batch_normalization_38[0][0]     
__________________________________________________________________________________________________
conv1d_39 (Conv1D)              (None, 15, 256)      327936      activation_35[0][0]              
__________________________________________________________________________________________________
batch_normalization_39 (BatchNo (None, 15, 256)      1024        conv1d_39[0][0]                  
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 15, 256)      0           batch_normalization_39[0][0]     
__________________________________________________________________________________________________
conv1d_40 (Conv1D)              (None, 15, 256)      327936      activation_36[0][0]              
__________________________________________________________________________________________________
batch_normalization_40 (BatchNo (None, 15, 256)      1024        conv1d_40[0][0]                  
__________________________________________________________________________________________________
add_12 (Add)                    (None, 15, 256)      0           activation_34[0][0]              
                                                                 batch_normalization_40[0][0]     
__________________________________________________________________________________________________
activation_37 (Activation)      (None, 15, 256)      0           add_12[0][0]                     
__________________________________________________________________________________________________
conv1d_41 (Conv1D)              (None, 15, 256)      327936      activation_37[0][0]              
__________________________________________________________________________________________________
batch_normalization_41 (BatchNo (None, 15, 256)      1024        conv1d_41[0][0]                  
__________________________________________________________________________________________________
activation_38 (Activation)      (None, 15, 256)      0           batch_normalization_41[0][0]     
__________________________________________________________________________________________________
conv1d_42 (Conv1D)              (None, 15, 256)      327936      activation_38[0][0]              
__________________________________________________________________________________________________
batch_normalization_42 (BatchNo (None, 15, 256)      1024        conv1d_42[0][0]                  
__________________________________________________________________________________________________
activation_39 (Activation)      (None, 15, 256)      0           batch_normalization_42[0][0]     
__________________________________________________________________________________________________
conv1d_43 (Conv1D)              (None, 15, 256)      327936      activation_39[0][0]              
__________________________________________________________________________________________________
batch_normalization_43 (BatchNo (None, 15, 256)      1024        conv1d_43[0][0]                  
__________________________________________________________________________________________________
add_13 (Add)                    (None, 15, 256)      0           activation_37[0][0]              
                                                                 batch_normalization_43[0][0]     
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 15, 256)      0           add_13[0][0]                     
__________________________________________________________________________________________________
conv1d_44 (Conv1D)              (None, 15, 512)      655872      activation_40[0][0]              
__________________________________________________________________________________________________
batch_normalization_44 (BatchNo (None, 15, 512)      2048        conv1d_44[0][0]                  
__________________________________________________________________________________________________
activation_41 (Activation)      (None, 15, 512)      0           batch_normalization_44[0][0]     
__________________________________________________________________________________________________
conv1d_45 (Conv1D)              (None, 15, 512)      1311232     activation_41[0][0]              
__________________________________________________________________________________________________
batch_normalization_45 (BatchNo (None, 15, 512)      2048        conv1d_45[0][0]                  
__________________________________________________________________________________________________
activation_42 (Activation)      (None, 15, 512)      0           batch_normalization_45[0][0]     
__________________________________________________________________________________________________
conv1d_47 (Conv1D)              (None, 15, 512)      655872      activation_40[0][0]              
__________________________________________________________________________________________________
conv1d_46 (Conv1D)              (None, 15, 512)      1311232     activation_42[0][0]              
__________________________________________________________________________________________________
batch_normalization_47 (BatchNo (None, 15, 512)      2048        conv1d_47[0][0]                  
__________________________________________________________________________________________________
batch_normalization_46 (BatchNo (None, 15, 512)      2048        conv1d_46[0][0]                  
__________________________________________________________________________________________________
add_14 (Add)                    (None, 15, 512)      0           batch_normalization_47[0][0]     
                                                                 batch_normalization_46[0][0]     
__________________________________________________________________________________________________
activation_43 (Activation)      (None, 15, 512)      0           add_14[0][0]                     
__________________________________________________________________________________________________
conv1d_48 (Conv1D)              (None, 15, 512)      1311232     activation_43[0][0]              
__________________________________________________________________________________________________
batch_normalization_48 (BatchNo (None, 15, 512)      2048        conv1d_48[0][0]                  
__________________________________________________________________________________________________
activation_44 (Activation)      (None, 15, 512)      0           batch_normalization_48[0][0]     
__________________________________________________________________________________________________
conv1d_49 (Conv1D)              (None, 15, 512)      1311232     activation_44[0][0]              
__________________________________________________________________________________________________
batch_normalization_49 (BatchNo (None, 15, 512)      2048        conv1d_49[0][0]                  
__________________________________________________________________________________________________
activation_45 (Activation)      (None, 15, 512)      0           batch_normalization_49[0][0]     
__________________________________________________________________________________________________
conv1d_50 (Conv1D)              (None, 15, 512)      1311232     activation_45[0][0]              
__________________________________________________________________________________________________
batch_normalization_50 (BatchNo (None, 15, 512)      2048        conv1d_50[0][0]                  
__________________________________________________________________________________________________
add_15 (Add)                    (None, 15, 512)      0           activation_43[0][0]              
                                                                 batch_normalization_50[0][0]     
__________________________________________________________________________________________________
activation_46 (Activation)      (None, 15, 512)      0           add_15[0][0]                     
__________________________________________________________________________________________________
conv1d_51 (Conv1D)              (None, 15, 512)      1311232     activation_46[0][0]              
__________________________________________________________________________________________________
batch_normalization_51 (BatchNo (None, 15, 512)      2048        conv1d_51[0][0]                  
__________________________________________________________________________________________________
activation_47 (Activation)      (None, 15, 512)      0           batch_normalization_51[0][0]     
__________________________________________________________________________________________________
conv1d_52 (Conv1D)              (None, 15, 512)      1311232     activation_47[0][0]              
__________________________________________________________________________________________________
batch_normalization_52 (BatchNo (None, 15, 512)      2048        conv1d_52[0][0]                  
__________________________________________________________________________________________________
activation_48 (Activation)      (None, 15, 512)      0           batch_normalization_52[0][0]     
__________________________________________________________________________________________________
conv1d_53 (Conv1D)              (None, 15, 512)      1311232     activation_48[0][0]              
__________________________________________________________________________________________________
batch_normalization_53 (BatchNo (None, 15, 512)      2048        conv1d_53[0][0]                  
__________________________________________________________________________________________________
add_16 (Add)                    (None, 15, 512)      0           activation_46[0][0]              
                                                                 batch_normalization_53[0][0]     
__________________________________________________________________________________________________
activation_49 (Activation)      (None, 15, 512)      0           add_16[0][0]                     
__________________________________________________________________________________________________
average_pooling1d_1 (AveragePoo (None, 15, 512)      0           activation_49[0][0]              
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 7680)         0           average_pooling1d_1[0][0]        
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         7865344     flatten_1[0][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 1024)         0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1024)         1049600     dropout_1[0][0]                  
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            1025        dense_2[0][0]                    
__________________________________________________________________________________________________
batch_normalization_54 (BatchNo (None, 1)            4           dense_3[0][0]                    
__________________________________________________________________________________________________
activation_50 (Activation)      (None, 1)            0           batch_normalization_54[0][0]     
==================================================================================================
Total params: 30,169,393
Trainable params: 27,893,187
Non-trainable params: 2,276,206
__________________________________________________________________________________________________
</code></pre>
</details>

## Performance

### Proposed Model

#### Statistical Model

1. Feature extracted by `CountVectorizer`

||Naive Bayes|Gaussian Bayes|Bernoulli Bayes|
|---|---|---|---|
|Accuracy|**0.7904**|0.6606|0.7540|
|F1 Score|**0.8357**|0.7689|0.8172|

2. Feature extracted by `TfidfVectorizer`

||Naive Bayes|Gaussian Bayes|Bernoulli Bayes|
|---|---|---|---|
|Accuracy|**0.7879**|0.6703|0.7540|
|F1 Score|**0.8362**|0.7708|0.8172|

#### Deep Learning Model

Feature extracted by word2vec.

||Simple-RNN|GRU|LSTM|BiLSTM|Attention|CNN-Static|CNN-MultiChannel|CNN-LSTM|Text-ResNet|
|---|---|---|---|---|---|---|---|---|---|
|Accuracy|0.7419|**0.7821**|0.7697|0.7430|0.7496|0.7736|0.7744|0.7380|0.7283|
|F1 Score|0.7773|**0.8216**|0.7945|0.7696|0.7857|0.8031|0.8073|0.7785|0.7535|
|Total params (M)|2.27|2.31|2.37|2.55|2.50|2.79|7.29|3.31|30.16|
|Trainable params (M)|0.02|0.06|0.12|0.30|0.25|0.54|0.54|1.05|27.89|
|Non-trainable params (M)|2.25|2.25|2.25|2.25|2.25|2.25|6.75|2.25|2.27|

#### Pre-trained Language Model

||BERT|ALBERT|DISTILBERT|
|---|---|---|---|
|Accuracy|**0.8543**|0.8005|0.8528|
|F1 Score|0.8806|0.8410|**0.8815**|

## Reference
1. [[link](https://arxiv.org/pdf/1606.01781.pdf)] Alexis Conneau, Very Deep Convolutional Networks for Text Classification 
2. [[link](https://anlp.jp/proceedings/annual_meeting/2018/pdf_dir/P12-2.pdf)] Lasguido Nio, Japanese Sentiment Classification Using Bidirectional Long Short-Term Memory Recurrent Neural Network 
3. [[link](https://www.scitepress.org/Papers/2017/61934/61934.pdf)] Minato Sato, Japanese Text Classification by Character-level Deep ConvNets and Transfer Learning 
4. [[link](https://www.aclweb.org/anthology/D14-1181.pdf)] Yoon Kim, Convolutional Neural Networks for Sentence Classification 
5. [[link](https://arxiv.org/pdf/1810.04805.pdf)] Jacob Devlin, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding 
6. [[link](https://arxiv.org/pdf/1909.11942.pdf)] Zhenzhong Lan, ALBERT: A Lite BERT for Self-supervised Learning of Language Representations 
7. [[link](https://arxiv.org/pdf/1910.01108.pdf)] Victor Sanh, DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter 
8. [[link](https://www.bioinf.jku.at/publications/older/2604.pdf)] Sepp Hochreiter, Long Short-Term Memory 
9. [[link](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)] Tomas Mikolov, Distributed representations of words and phrases and their compositionality 
10. [[link](https://www.aclweb.org/anthology/E17-1096.pdf)] Amr El-Desoky Mousa, Contextual bidirectional long short-term memory recurrent neural network language models: A generative approach to sentiment analysis 
11. [[link](https://dl.acm.org/doi/pdf/10.1145/2766462.2767830)] Aliaksei Severyn, Twitter sentiment analysis with deep convolutional neural networks 
12. [[link](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)] Nitish Srivastava, Dropout: A Simple Way to Prevent Neural Networks from Overfitting 
13. [[link](https://arxiv.org/pdf/1502.03167.pdf)] Sergey Ioffe, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift 
14. [[link](https://arxiv.org/pdf/1602.07868.pdf)] Tim Salimans, Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks 
15. [[link](https://arxiv.org/pdf/1706.03762.pdf)] Ashish Vaswani, Attention Is All You Need 
16. [[link](https://arxiv.org/pdf/1906.08237.pdf)] Zhilin Yang, XLNet: Generalized Autoregressive Pretraining for Language Understanding 
17. [[link](https://arxiv.org/pdf/1904.08398.pdf)] Ashutosh Adhikari, DocBERT: BERT for Document Classification 

## Future Roadmap
It is completely possible to use only raw text as input for making predictions. The most important thing is to automatically extract the relevant features from this raw source of reviews data. Although the models don't perform well and need more improvement, I have done a practise with a full harvest.

Text classifier is a meat-and-potatoes issue for most sentiment analysis task, and there are still many things can be done on this task. In future works, I might construct a multi-class text classifier to separate customers' reviews into different issue types. (e.g. Function, UI, Crash, Truncate, Subscription, Server, Enhancement, etc), in order to tackle each consumer's problem more efficiently and effectively.

