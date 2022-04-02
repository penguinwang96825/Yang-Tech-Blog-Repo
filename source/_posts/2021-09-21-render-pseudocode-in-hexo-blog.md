---
title: Render Pseudocode in Hexo Blog
top: false
cover: false
toc: true
mathjax: true
date: 2021-09-21 16:08:25
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/21/2021-09-21-render-pseudocode-in-hexo-blog/wallhaven-2839em.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/21/2021-09-21-render-pseudocode-in-hexo-blog/wallhaven-2839em.jpg?raw=true
summary: pseudocode.js is a JavaScript library that typesets pseudocode beautifully to HTML. When I was trying to render the pseudocode in my tech blog, I struggled a lot. I tried loads of methods, but none of them worked. Finally, this one worked out, so I'd like to document it for future reference.
categories: Script
tags:
	- Hexo
	- Javascript
	- Pseudocode
---

# Introduction

`pseudocode.js` is a JavaScript library that typesets pseudocode beautifully to HTML. When I was trying to render the pseudocode in my tech blog, I struggled a lot. I tried loads of methods, but none of them worked. Finally, this one worked out, so I'd like to document it for future reference.

# Quick Start

## Step One

Render math formulas using `MathJax`. Include the following in the `layout/_partial/head.ejs`

```javascript
<script src='http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default'>
</script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [['$','$'], ['\\(','\\)']],
            displayMath: [['$$','$$'], ['\\[','\\]']],
            processEscapes: true,
            processEnvironments: true,
        }
    });
</script>
```

## Step Two

Grab `pseudocode.js` by including the following in the `layout/_partial/head.ejs` of the page.

```javascript
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pseudocode@latest/build/pseudocode.min.css">
<script src="https://cdn.jsdelivr.net/npm/pseudocode@latest/build/pseudocode.min.js">
</script>
```

## Step Three

Write the pseudocode inside a `<pre>`. Here is an example that illustrates a quicksort algorithm.

```
<pre id="pseudocode" style="display:hidden;">
    \begin{algorithm}
    \caption{Quicksort}
    \begin{algorithmic}
    \PROCEDURE{Quicksort}{$A, p, r$}
        \IF{$p < r$} 
            \STATE $q = $ \CALL{Partition}{$A, p, r$}
            \STATE \CALL{Quicksort}{$A, p, q - 1$}
            \STATE \CALL{Quicksort}{$A, q + 1, r$}
        \ENDIF
    \ENDPROCEDURE
    \PROCEDURE{Partition}{$A, p, r$}
        \STATE $x = A[r]$
        \STATE $i = p - 1$
        \FOR{$j = p$ \TO $r - 1$}
            \IF{$A[j] < x$}
                \STATE $i = i + 1$
                \STATE exchange
                $A[i]$ with $A[j]$
            \ENDIF
            \STATE exchange $A[i]$ with $A[r]$
        \ENDFOR
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}
</pre>
```

<pre id="pseudocode" style="display:hidden;">
    \begin{algorithm}
    \caption{Quicksort}
    \begin{algorithmic}
    \PROCEDURE{Quicksort}{$A, p, r$}
        \IF{$p < r$} 
            \STATE $q = $ \CALL{Partition}{$A, p, r$}
            \STATE \CALL{Quicksort}{$A, p, q - 1$}
            \STATE \CALL{Quicksort}{$A, q + 1, r$}
        \ENDIF
    \ENDPROCEDURE
    \PROCEDURE{Partition}{$A, p, r$}
        \STATE $x = A[r]$
        \STATE $i = p - 1$
        \FOR{$j = p$ \TO $r - 1$}
            \IF{$A[j] < x$}
                \STATE $i = i + 1$
                \STATE exchange
                $A[i]$ with $A[j]$
            \ENDIF
            \STATE exchange $A[i]$ with $A[r]$
        \ENDFOR
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}
</pre>

## Step Four

Render the element using `pseudocode.js`. Insert the following Javascript snippet at the end of the document, which is the file `layout/layout.ejs`.

```javascript
<script>
    pseudocode.renderElement(document.getElementById("pseudocode"));
</script>
```

# Grammar

Commands for typesetting algorithms must be enclosed in an `algorithmic` environment.

```
\begin{algorithmic}
	# A precondition is optional
	\REQUIRE <text>
	# A postcondition is optional
	\ENSURE <text>
	# An input is optional
	\INPUT <text>
	# An output is optional
	\OUTPUT <text>
	# The body of your code is a <block>
	\STATE ...
\end{algorithmic}
```

`<block>` can include zero or more `<statement>`, `<control>`, `<comment>` and `<function>`.

```
# A <statement> can be:
\STATE <text>
\RETURN <text>
\PRINT <text>

# A <control> can be:
# A conditional
\IF{<condition>}
    <block>
\ELIF{<condition>}
    <block>
\ELSE
    <block>
\ENDIF
# Or a loop: \WHILE, \FOR or \FORALL
\WHILE{<condition>}
    <block>
\ENDWHILE
# Or a repeat: \REPEAT <block> \UNTIL{<cond>}
\REPEAT
    <block>
\UNTIL{<cond>}

# A <function> can by defined by either \FUNCTION or \PROCEDURE
# Both are exactly the same
\FUNCTION{<name>}{<params>}
    <block> 
\ENDFUNCTION

# A <comment> is:
\COMMENT{<text>}
```

A `<text>` (or `<condition>`) can include the following.

```
# Normal characters
Hello world
# Escaped characters
\\, \{, \}, \$, \&, \#, \% and \_
# Math formula
$i \gets i + 1$
# Function call
\CALL{<func>}{<args>}
# Keywords
\AND, \OR, \XOR, \NOT, \TO, \DOWNTO, \TRUE, \FALSE
# LaTeX's sizing commands
\tiny, \scriptsize, \footnotesize, \small \normalsize, \large, \Large, \LARGE, 
\huge, \HUGE
# LaTeX's font declarations
\rmfamily, \sffamily, \ttfamily
\upshape, \itshape, \slshape, \scshape
\bfseries, \mdseries, \lfseries
# LaTeX's font commands
\textnormal{<text>}, \textrm{<text>}, \textsf{<text>}, \texttt{<text>}
\textup{<text>}, \textit{<text>}, \textsl{<text>}, \textsc{<text>}
\uppercase{<text>}, \lowercase{<text>}
\textbf, \textmd, \textlf
# And it's possible to group text with braces
normal text {\small the size gets smaller} back to normal again
```

# References

1. https://github.com/SaswatPadhi/pseudocode.js