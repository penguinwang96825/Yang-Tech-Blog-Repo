---
title: MeCab and CaboCha for Japanese
top: false
cover: false
toc: true
mathjax: true
date: 2021-04-27 00:20:40
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/04/27/2021-04-27-mecab-and-cabocha-for-japanese/wallhaven-j3xm5w.png?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/04/27/2021-04-27-mecab-and-cabocha-for-japanese/wallhaven-j3xm5w.png?raw=true
summary: In Python, there are several choices of modules for morphological analysis. There are several types of kuromoji such as Janome, Juman, MeCab, and Esanpy, but this time we will use MeCab, which is said to be relatively fast and accurate.
tags:
	- Python
	- Japanese
	- Text Processing
categories: NLP
---

# Introduction

In Python, there are several choices of modules for morphological analysis. There are several types of kuromoji such as Janome, Juman, MeCab, and Esanpy, but this time we will use MeCab, which is said to be relatively fast and accurate.

# MeCab

[MeCab](http://taku910.github.io/mecab/) is an open source morphological analysis engine developed through a joint research unit project between the Graduate School of Informatics, Kyoto University and the Communication Science Laboratories of Nippon Telegraph and Telephone Corporation. By the way, University of Sheffield, where I am studying at, had some kind of collaboration with NAIST during the 2012 Olympics in the UK, and the UoS helped NAIST produce a speech-to-speech translation app called VoiceTra for the event. MeCab is designed to be language, dictionary, and corpus independent.

Best Practices for Installing MeCab and CaboCha on Google Colab (refer from [here](https://gist.github.com/tomowarkar/021580fa52781ed0b0d913f46c8bb7e5)):

```python
%%bash
# mecabとその依存関係のインストール
apt-get install mecab swig libmecab-dev mecab-ipadic-utf8
# mecab-pythonのインストール
pip install mecab-python3

# crfppダウンロード(cabochaの依存関係)
curl -sL -o CRF++-0.58.tar.gz "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ"
tar -zxf CRF++-0.58.tar.gz
# crfppインストール
cd CRF++-0.58
./configure && make && make install && ldconfig
cd ..

# cabochaダウンロード
url="https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU"
curl -sc /tmp/cookie ${url} >/dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -sLb /tmp/cookie ${url}"&confirm=${code}" -o cabocha-0.69.tar.bz2
tar -jxf cabocha-0.69.tar.bz2
# cabochaインストール
cd cabocha-0.69
./configure -with-charset=utf-8 && make && make check && make install && ldconfig
# cabocha-pythonのインストール
pip install python/
cd ..
```

Check the version of MeCab:

```python
%%bash
mecab -v
pip show mecab-python3 | grep -e Name -e Version && echo
```

## Installation

You can install via `pip install mecab` or `python -m pip install mecab`. If you use Python 64-bit on Windows, [MeCab 64-bit binary](https://github.com/ikegami-yukino/mecab/releases) is required. If you encountered the error such as MeCab_wrap.cxx:178:11: fatal error: 'Python.h' file not found, please try the following command:

```bash
CPLUS_INCLUDE_PATH=`python-config --prefix`/Headers:$CPLUS_INCLUDE_PATH pip install mecab
```

If you are a mac user, you can install MeCab and the dictionary more easier through this way.

```bash
brew install mecab
brew install mecab-ipadic
```

Next, install `SWIG`. We have to install this library, otherwise, there would be errors when installing mecab-python3.

```bash
brew install swig
swig -version
```

Finally, install the mecab-python3.

```bash
pip install mecab-python3
```

If you want to add a custom user dictionary, you can take a look at this [article](https://towardsdatascience.com/mecab-usage-and-add-user-dictionary-to-mecab-9ee58966fc6).

## MeCab Example

Now let's try morphological analysis in Python. The input text is "色んな曲が聴けるのでとても良いです。". The code looks like the following.

```python
import MeCab
wakati = MeCab.Tagger()
print(wakati.parse("色んな曲が聴けるのでとても良いです。"))
```

If we run this through python, 

```
色んな	イロンナ	イロンナ	色んな	連体詞			0
曲	キョク	キョク	曲	名詞-普通名詞-一般			0,1,2
が	ガ	ガ	が	助詞-格助詞			
聴ける	キケル	キク	聞く	動詞-一般	下一段-カ行	連体形-一般	0
の	ノ	ノ	の	助詞-準体助詞			
で	デ	ダ	だ	助動詞	助動詞-ダ	連用形-一般	
とても	トテモ	トテモ	迚も	副詞			0
良い	ヨイ	ヨイ	良い	形容詞-非自立可能	形容詞	終止形-一般	1
です	デス	デス	です	助動詞	助動詞-デス	終止形-一般	
。			。	補助記号-句点			
EOS
```

By the way, we can choose different output mode for MeCab.
- `mecabrc`: default
- `-Ochasen`: ChaSen compatible format
- `-Owakati`: output only shared writing
- `-Oyomi`: outputs read-only

These are the four modes available. You may want to try different ones. Moreover, you will be able to analyze the text in another way.

```python
sentence = "色んな曲が聴けるのでとても良いです。"
node = wakati.parseToNode(sentence)
tokens, taggings = [], []
while node:
	print(node.surface, "\t", node.feature)
	tokens.append(node.surface)
	taggings.append(node.feature.split(",")[0])
	node = node.next
```

The `tokens` will be looked like the following:

> ['', '色んな', '曲', 'が', '聴ける', 'の', 'で', 'とても', '良い', 'です', '。', '']

The `taggings` will be looked like the following:

> ['BOS/EOS', '連体詞', '名詞', '助詞', '動詞', '助詞', '助動詞', '副詞', '形容詞', '助動詞', '補助記号', 'BOS/EOS']

Word segmentation and part-of-speech (POS) tagging are considered fundamental steps for high-level natural language processing tasks such as parsing, machine translation, and information extraction.

I have created a module to list the output of MeCab. It looks like the following.

```python
class Analyser:

	def __init__(self):
		self.tagger = MeCab.Tagger()

	def analyse(self, text):
		self.tagger.parse('')
		node = self.tagger.parseToNode(text)
		results = []
		while node:
			word = node.surface
			wclass = node.feature.split(',')
			if wclass[0] != u'BOS/EOS':
				if wclass[6] == None:
					results.append((word,wclass[0],wclass[1],wclass[2],""))
				else:
					results.append((word,wclass[0],wclass[1],wclass[2],wclass[6]))
			node = node.next
		return results
```

It can be used like this.

```python
sentence = "色んな曲が聴けるのでとても良いです。"
analyser = Analyser()
analyser.analyse(sentence)
```

---

```
[('色んな', '連体詞', '*', '*', 'イロンナ'),
 ('曲', '名詞', '普通名詞', '一般', 'キョク'),
 ('が', '助詞', '格助詞', '*', 'ガ'),
 ('聴ける', '動詞', '一般', '*', 'キク'),
 ('の', '助詞', '準体助詞', '*', 'ノ'),
 ('で', '助動詞', '*', '*', 'ダ'),
 ('とても', '副詞', '*', '*', 'トテモ'),
 ('良い', '形容詞', '非自立可能', '*', 'ヨイ'),
 ('です', '助動詞', '*', '*', 'デス'),
 ('。', '補助記号', '句点', '*', '')]
```

### Rough Explanation

```python
tagger.parse('')
```

By doing this `tagger.parse('')` before passing the data to the parser, you can avoid `UnicodeDecodeError`. I don't know the exact reason, but I think it is probably because once the `tagger.parse('')` is inserted, it is initialized with the standard character encoding used in the program.

```python
node = tagger.parseToNode(text)
```

Assigning analysis results with surface (word) and feature (part-of-speech information) to node. The data in the `node.feature` part is:

> 表層形, 左文脈ID, 右文脈ID, コスト, 品詞, 品詞細分類1, 品詞細分類2, 品詞細分類3, 活用型, 活用形, 原形, 読み, 発音

> Surface type, left context ID, right context ID, cost, part of speech, sub POS 1, sub-POS 2, sub-POS 3, conjugation type, conjugation form, original form, ruby, pronunciation

It is split by "," and assigned to `wclass` as an array. 

```python
if wclass[6] == None:
	results.append((word,wclass[0],wclass[1],wclass[2],""))
else:
	results.append((word,wclass[0],wclass[1],wclass[2],wclass[6]))
```

The content was in the form of (surface form, parts of speech, part-of-speech subdivision 1, part-of-speech subdivision 2, original form). If you need other data, please change the wclass[] part accordingly.

# CaboCha

[CaboCha](http://taku910.github.io/cabocha/) is another Japanese Dependency Structure Analyser. It is a Japanese Kanji analyzer based on Support Vector Machines.

## Installation

```bash
%%bash
git clone https://github.com/kenkov/cabocha
pip install cabocha/
```

Check the version of CaboCha:

```python
%%bash
cabocha -v && echo
pip show cabocha-python | grep -e Name -e Version
```

## Usage

Simple example:

```python
import CaboCha
cp = CaboCha.Parser()
print(cp.parseToString("隣の客はよく柿食う客だ"))
```

---

```
隣の-D        
  客は-------D
    よく---D |
        柿-D |
        食う-D
          客だ
EOS
```

Or you can use it this way.

```python
from cabocha.analyzer import CaboChaAnalyzer
analyzer = CaboChaAnalyzer()
tree = analyzer.parse("日本語の形態素解析はすごいです。")
tokens = []
for chunk in tree:
	for token in chunk:
		tokens.append(token.surface)
```

---

```
['日本語', 'の', '形態素', '解析', 'は', 'すごい', 'です', '。']
```

# Conclusion

By using MeCab, I can always create the application for the task regarding to Japanese tasks. It's not quite ready for practical use, but it's still a lot of work to get it up and running. The only thing left to do is to grow the application with imagination and effort. I put the colab notebook over [here](https://colab.research.google.com/drive/1zbJUoDl8PsiRooRfF8f001l1O1BigEkk?usp=sharing), please feel free to read it.

# Reference

1. https://snyk.io/advisor/python/mecab
2. https://qiita.com/menon/items/f041b7c46543f38f78f7
3. https://gist.github.com/tomowarkar/021580fa52781ed0b0d913f46c8bb7e5
4. https://qiita.com/menon/items/2b5ad487a98882289567
5. https://qiita.com/ryo--kato/items/f39dc084e0f2cec1cc0d
6. https://rstudio-pubs-static.s3.amazonaws.com/462850_98582068058d4191a70b7246d2ceee29.html
7. https://github.com/taku910/cabocha
8. https://taku910.github.io/cabocha/