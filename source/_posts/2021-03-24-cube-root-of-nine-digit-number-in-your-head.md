---
title: Cube Root of 9 Digit Number in Your Head
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-24 00:31:53
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/24/2021-03-24-cube-root-of-nine-digit-number-in-your-head/wallhaven-2eroxm.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/24/2021-03-24-cube-root-of-nine-digit-number-in-your-head/wallhaven-2eroxm.jpg?raw=true
summary: In this article, I will show you how to find the cube root of 9 digit number. This is a very effective way of finding cube root of such high numbers. If you practise rigorously, you can do it in 10 seconds!
tags:
	- Mathematics
	- Mental Arithmetic
categories: Mathematics
---

# Introduction

In this article, I will show you how to find the cube root of 9 digit number. This is a very effective way of finding cube root of such high numbers. If you practise rigorously, you can do it in 10 seconds!

# Cube Roots up to 6 Digits

To do cube roots of digit numbers, you are going to have to be familiar with these first 10 cubes.

| {% mathjax %} n {% endmathjax %} | {% mathjax %} n^{3} {% endmathjax %} |
| --- | --- |
| 1 | 1 |
| 2 | 8 |
| 3 | 27 |
| 4 | 64 |
| 5 | 125 |
| 6 | 256 |
| 7 | 343 |
| 8 | 512 |
| 9 | 729 |
| 10 | 1000 |

If {% mathjax %} 10n \leq x \leq 10(n+1) {% endmathjax %}, then {% mathjax %} 1000n^3 \leq x^3 \leq 1000(n+1)^3 {% endmathjax %}, and therefore {% mathjax %} n^3 \leq \frac{1000}{x^3} \leq (n+1)^3 {% endmathjax %}. Since {% mathjax %} n^{3} {% endmathjax %} and {% mathjax %} (n+1)^{3} {% endmathjax %} are integers, you can ignore the fractional part of {% mathjax %} \frac{x^3}{1000} {% endmathjax %}, which is essentially what you do when you drop the last three digits of {% mathjax %} x {% endmathjax %}. Also, you can know that for a 6 digits number, and it is also a perfect cube ({% mathjax %} x^{3} {% endmathjax %} where x is an integer), then {% mathjax %} x {% endmathjax %} must be a two digits number.

We will follow these steps to find the cube root of a 6 digit number:
1. Find a number {% mathjax %} F {% endmathjax %}, which {% mathjax %} F^{3} {% endmathjax %} is greater than first three digits, and {% mathjax %} (F+1)^{3} {% endmathjax %} less than first three digits.
2. Find a number {% mathjax %} S {% endmathjax %}, which the last digit of {% mathjax %} S^{3} {% endmathjax %} is equal to the last digit of the 6 digits number.

Take 300763 for example, 

1. The first three digits is 300, you can see that we have to choose {% mathjax %} F = 6 {% endmathjax %}, because {% mathjax %} 6^3 \leq 300 \leq 7^3 {% endmathjax %}.
2. The last three digits is 763, you can see that we have to choose {% mathjax %} S = 7 {% endmathjax %}, because the last digit of {% mathjax %} 7^3 {% endmathjax %} is 3, which is the same digit as in 763.

# Cube Roots up to 9 Digits

Similarly, you can find out that a 9 digits number must be a three digits number cubed. 

We will follow the following steps to find the cube root of a 9 digits number.
1. Divide the 9 digits number into three parts, which is first three digits, middle three digits, and last three digits. Also, we assume the cube root of this 9 digits number is "FST" (F denotes first, S denotes second, and T denotes third).
2. You can find "F" and "T" using the same method as in finding the cube root of a 6 digits number.
3. For finding "T", first, you need to calculate the last three digits minus T, and keep the second digit as {% mathjax %} x {% endmathjax %}. Second, compute {% mathjax %} 3 T^2 S = x {% endmathjax %}, and get the potential candidates for S. Finally, if the potential candidates is not only one, use digital sum method to find S.

I will explain this using two examples. Let's see what's the cube root of 196122941 first.

1. Divide 196122941 into three parts, you will have 196, 122, and 941.
2. You can figure out that {% mathjax %} F = 5 {% endmathjax %} and {% mathjax %} T = 1 {% endmathjax %}.
3. First, we compute {% mathjax %} 941 - T^3 = 940 {% endmathjax %}, and keep the second digit, which is 4. Next, compute {% mathjax %} 3 T^3 S = 4 {% endmathjax %}, you will have {% mathjax %} 3S = 4 {% endmathjax %}. In this situation, you will have to see what number multiplied by 3, and its last digit is 4. The answer is 8, so you get {% mathjax %} S = 8 {% endmathjax %}.

Awesome! Let's see another example, the cube root of 392223168.

1. Divide 392223168 into three parts, 392, 223, and 168.
2. You can figure out that {% mathjax %} F = 7 {% endmathjax %} and {% mathjax %} T = 2 {% endmathjax %}.
3. First, we compute {% mathjax %} 168 - T^3 = 160 {% endmathjax %}, and keep the second digit, which is 6. Next, compute {% mathjax %} 3 T^3 S = 6 {% endmathjax %}, you will have {% mathjax %} 12S = 6 {% endmathjax %}. In this situation, you will have to see what number multiplied by 3, and its last digit is 4. The answer is 3 or 8, that means there's two potential number 732 or 782, so you have to use digital sum method to find S. What digital sum method does is to sum up all the digits in base 10. In this case, {% mathjax %} 3+9+2+2+2+3+1+6+8=36=3+6=9 {% endmathjax %}, so the digital sum of 392223168 is 9. Compute the same for 732, you get the difital sum of 732 is 3, and {% mathjax %} 3^3 = 27 = 2+7 = 9 {% endmathjax %}; as for 782, you get the digital number of 782 is {% mathjax %} 8 {% endmathjax %}, and {% mathjax %} 8^3 = 512 = 5+1+2 = 8 {% endmathjax %}. Finally, 732 matches, the answer is 732!

# Conclusion

Hope this will help you to find out the cube root of a number quicker! Cheers!