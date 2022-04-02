---
title: Prevent Colab from Disconnecting
date: 2021-02-06 09:25:00
author: Yang Wang
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/02/06/2021-02-06-prevent-colab-from-disconnecting/kai-wenzel.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/02/06/2021-02-06-prevent-colab-from-disconnecting/colab.png?raw=true
summary: Google Colab notebooks have an idle timeout of 90 minutes and absolute timeout of 12 hours. This means, if user does not interact with his Google Colab notebook for more than 90 minutes, its instance is automatically terminated. Also, maximum lifetime of a Colab instance is 12 hours.
categories: Script
tags:
  - Colab
  - JavaScript
---

Google Colab notebooks have an idle timeout of 90 minutes and absolute timeout of 12 hours. This means, if user does not interact with his Google Colab notebook for more than 90 minutes, its instance is automatically terminated. Also, maximum lifetime of a Colab instance is 12 hours.

# Solution

It is wonderful that Google colab can use GPU, TPU, and other computing resources for artificial intelligence calculation for free, but the calculation page will be automatically dropped after a period of no operation, and the previous training data will be lost, which is disappointing.

Finally found a way to keep it from going offline automatically, with a JavaScript program that automatically clicks the connect button.

**Step 1**: Press the shortcut keys `CTRL + SHIFT + I` and select `Console`.

**Step 2**: Copy and paste the below code, and hit `ENTER`, the program will be ready to run.

```javascript
function ClickConnect(){
  console.log("Working"); 
  document
    .querySelector("#top-toolbar > colab-connect-button")
    .shadowRoot
    .querySelector("#connect")
    .click()
}
 
setInterval(ClickConnect, 5*60000)
```

If you still have problems running it or want to get more methods, you can refer [here](https://stackoverflow.com/questions/57113226/how-to-prevent-google-colab-from-disconnecting).