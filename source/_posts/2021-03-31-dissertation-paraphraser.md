---
title: Dissertation Paraphraser
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-31 12:40:40
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/31/2021-03-31-dissertation-paraphraser/wallhaven-x1xeoo.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/31/2021-03-31-dissertation-paraphraser/wallhaven-x1xeoo.jpg?raw=true
summary: Paraphrasing and summarizing are vital so that you essay doesn't become one long quote of other academics' work. To paraphrase a piece of text is to write it in your own words. In this article, I will show you how I make an app that will help me rephrase the sentence I need.
tags:
    - Python
    - Flask
    - GUI
    - NLP
    - Pegasus
categories: NLP
---

# Introduction

Paraphrasing and summarizing are vital so that you essay doesn't become one long quote of other academics' work. To paraphrase a piece of text is to write it in your own words. In this article, I will show you how I make an app that will help me rephrase the sentence I need.

# Quickstart

Create desktop applications with `Flask` and `flaskwebgui`.

```bash
pip install flaskwebgui
```

Load the libraries we need.

```python
import os
import torch
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from flask import Flask, request, render_template
from flaskwebgui import FlaskUI
```

# Paraphrase NLP

I use a pre-trained model called [PEGASUS](https://github.com/google-research/pegasus), proposed by Google in 2020. It uses self-supervised objective Gap Sentences Generation (GSG) to train a transformer encoder-decoder model.

Also, I use huggingface library to get the fine-tuned model trained by [tuner007](https://huggingface.co/tuner007).

```python
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    global model
    global tokenizer
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text], 
                      truncation=True, 
                      padding='longest', 
                      max_length=60, 
                      return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, 
                                max_length=60, 
                                num_beams=num_beams, 
                                num_return_sequences=num_return_sequences, 
                                temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, 
                                      skip_special_tokens=True)
    return tgt_text
```

# Python App GUI

Next, create a file named `app.py`.

```bash
touch app.py
```

Put the following code inside of it.

```python
import os
import torch
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from flask import Flask, request, render_template
from flaskwebgui import FlaskUI


app = Flask(__name__)
ui = FlaskUI(app, width=800, height=1000)
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    elif request.method == "POST":
        context = request.form["paragraph"]
        num_beams = 10
        num_return_sequences = 10
        tgt_text = get_response(context, num_return_sequences, num_beams)
        return render_template('index.html', src_text=context, tgt_text=tgt_text)


@app.before_first_request
def load_model():
    global model
    global tokenizer
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text], 
                      truncation=True, 
                      padding='longest', 
                      max_length=60, 
                      return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, 
                                max_length=60, 
                                num_beams=num_beams, 
                                num_return_sequences=num_return_sequences, 
                                temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, 
                                      skip_special_tokens=True)
    return tgt_text


if __name__ == '__main__':
    ui.run()
```

# Frontend Interface

Create two folders, `static/css` and `templates`, and create `style.css` in `static/css` folder, and create html files `base.html` and `index.html` in `templates` folder.

1. `base.html`

```html
<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <!-- Semantic UI -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css">
    <script src="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.js"></script>

    <title>{% block title %} {% endblock %}</title>
  </head>
  <body>
    <nav class="navbar navbar-expand-md navbar-light bg-light">
        <a class="navbar-brand" href="{{ url_for('index')}}">Paraphraser</a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
            <li class="nav-item active">
                <a class="nav-link" href="#">About</a>
            </li>
            </ul>
        </div>
    </nav>
    <div class="container">
        {% block content %} {% endblock %}
    </div>

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
  </body>
</html>
```

2. `index.html`

```html
{% extends 'base.html' %}

{% block content %}
    <h1>Welcome to Paraphraser!</h1>
    <div class="form-group green-border-focus">
    <form method="POST">
        <div class="form-group">
            <textarea class="form-control" id="exampleFormControlTextarea5" placeholder="Your sentence here..." value=" " rows="10" cols="50" name="paragraph"></textarea>
        </div>
        <div class="form-group">
            <input class="btn btn-primary" type="submit" name="Submit" value="Submit">
        </div>
    </form>
    </div>
    <div>
        {% if src_text is not none %}
            <h3>Original Sentence: </h3>
            <p>{{ src_text }}</p>
            <br>
        {% endif %}
    </div>
    <div>
        {% if tgt_text is not none %}
            <h3>Paraphrased Sentences: </h3>
            <ul>
                {% for text in tgt_text %}
                    <li>{{ text }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </div>
{% endblock %}
```

3. `style.css`

```css
h1 {
    border: 2px #eee solid;
    color: brown;
    text-align: center;
    padding: 30px;
}

textarea {
    border: 1px solid #ba68c8;
}
.form-control:focus {
    border: 1px solid #ba68c8;
    box-shadow: 0 0 0 0.2rem rgba(186, 104, 200, .25);
}

.green-border-focus .form-control:focus {
    border: 1px solid #8bc34a;
    box-shadow: 0 0 0 0.2rem rgba(139, 195, 74, .25);
}
```

Run the code!

```bash
python app.py
```

{% asset_img demo.gif %}

# Conclusion

Check your paraphrasing for grammar and plagiarism. It's fast and easy!