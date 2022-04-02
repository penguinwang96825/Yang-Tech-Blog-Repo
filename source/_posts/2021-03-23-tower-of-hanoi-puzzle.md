---
title: Tower of Hanoi Puzzle
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-23 14:32:12
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/23/2021-03-23-tower-of-hanoi-puzzle/wallhaven-vm6mql.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/23/2021-03-23-tower-of-hanoi-puzzle/wallhaven-vm6mql.jpg?raw=true
summary: Tower of Hanoi is a mathematical puzzle where we have three rods and n disks. The objective is to move the entire stack to another rod, and follow some simple rules.
tags: 
    - Mathematics
    - Python
    - JavaScript
categories: Mathematics
---

# Introduction

Tower of Hanoi, invented by E. Lucas in 1883, is a mathematical puzzle where we have three rods and n disks. The objective is to move the entire stack to another rod, and follow some simple rules below:

1. Only one disk can be moved at a time.
2. A disk can only be moved if it is the uppermost disk on a stack.
3. No disk may be placed on top of a smaller disk.

{% asset_img hanoi.gif source: https://mathworld.wolfram.com/TowerofHanoi.html %}

# Strategy

Let's assume the left rod called "source", the middle rod called "dest", and the right rod called "aux." We can use a recursive solution for Tower of Hanoi. We assume that the number of moves {% mathjax %} h_{n} {% endmathjax %} required to solve the puzzle of n disks. First move the top disk, let's call it disk 1, from source to dest. If there's only one disk, then it is done. Otherwise, we should move n-1 disks from source to aux, which means that we will have {% mathjax %} h_{n-1} {% endmathjax %} moves. Next, move the nth disk from source to dest, thich means there's one move. Finally, move n-1 disks from aux to dest, which means that we will have {% mathjax %} h_{n-1} {% endmathjax %} moves. To conclude, we have {% mathjax %} 2 h_{n-1} + 1 {% endmathjax %} in total. Given by the recurrence relation, 

<div style="display: flex;justify-content: center;">
    {% mathjax %} h_{n} = 2 h_{n-1} + 1 {% endmathjax %} 
</div>

We also have {% mathjax %} h_{1} = 1 {% endmathjax %}. Solving this gives

<div style="display: flex;justify-content: center;">
    {% mathjax %} h_{n} = 2^{n} - 1 {% endmathjax %} 
</div>

which is also known as the [Mersenne numbers](https://mathworld.wolfram.com/MersenneNumber.html).

{% asset_img wolfram.gif source: https://mathworld.wolfram.com/TowerofHanoi.html %}

# Implementation

Here's how the Python code looks:

```python
def tower_of_hanoi(n, source, dest, aux):
    # Base Case
    if n == 1:
        print(f"Move disk 1 from {source} to {dest}", end="\n")
        return
    # Move (n-1) disks from source to aux.
    tower_of_hanoi(n - 1, source, aux, dest)
    # Move nth disk from source to dest.
    print(f"Move disk {n} from {source} to {dest}", end="\n")
    # Move (n-1) disks from aux to dest.
    tower_of_hanoi(n - 1, aux, dest, source)

tower_of_hanoi(5, "source", "dest", "aux")
```

Here's how the JavaScript code looks:

```javascript
function tower_of_hanoi(n, source, dest, aux) {
  if (n == 1) {
    // base case of 1 disk, we know how to solve that
    document.write("Move disk 1 from " + source + " to " + dest + ".<br/>");
  } else {
    // Move (n-1) disks from source to aux.
    tower_of_hanoi(n - 1, source, aux, dest);
    // now move the last disk
    document.write("Move disk " + n  + " from " + source + " to " + dest + ".<br/>");
    // Move (n-1) disks from aux to dest.
    tower_of_hanoi(n - 1, aux, dest, source);
  }
}
```

# Conclusion

The Tower of Hanoi puzzle is a great example of how recursion can more easily solve a problem. Recursion can be used for a variety of tasks including computations, sorting values, and quickly finding a value in a sorted list. The ability to break a problem into smaller pieces (Divide-and-conquer Algorithm) is a valuable skill for computer programming.

## References

1. https://mathworld.wolfram.com/TowerofHanoi.html
2. https://www.instructables.com/Write-Code-to-Solve-the-Tower-of-Hanoi-Puzzle/