Zubr Lisp
==================

ZubrLisp is a small (and somewhat hacky) python/cython implementation of Lisp based on Peter Norvig's ``lispy`` interpreter (in some cases, his original code is used directly).

It is mostly used to define and run macros that can manipulate semantic parse trees in various ways, and for implementing logical theories.

Use at your own risk (it is not well tested, and shouldn't be used for anything too serious).

Using the REPL
------------------

To run ZubrLisp, there is a top-level script in the zubr source distribution called ``zubr_lisp``, which will put you in a REPL:

```lisp
zubr >>> (+ 1 2) 
=> 3
zubr >>> (begin (define x 20) (if (= 19 19) (+ x 100) (+ x 50))
=> 120
zubr >>> (define myfun (lambda (x) (cond ;; cond special form
                        ((= x 10) (- x 1))
                        ((= x 12) (+ x 1))
                        0)))
=> None
zubr >>> (myfun 10)
=> 9
zubr >>> (define-macro simple-def
                      (lambda (name args body)
                      `(define ,name (lambda ,args ,body))))
=> None
zubr >>> (simple-def newfun (x) (+ x 20))
=> None
zubr >>> (newfun 20)
=> 40
zubr >>> parse (simple-def newer (x y) (+ x y)) ;; macro expansion 
=> [u'define', u'newer', [u'lambda', [u'x', u'y'], [u'+', u'x', u'y']]] 
zubr >>> (define my-dict (dict (zip '(1 2 3) '("one" "two" "three"))))
=> None
zubr >>> my-dict
=> {1: u'one', 2: u'two', 3: u'three'} 
zubr >>> (get-val my-dict 3)
=> three
zubr >>> (vals (make-dict '(1 2 3) '("one" "two" "three"))) 
["one", "two", "three"]
zubr >>> (new-record my-record ;; a record abstraction
            (first-attr 10)
            (second-attr (+ 10 20))
            (third-attr (lambda (x) (+ x 13))))
=> None
zubr >>> (gattr first-attr my-record)
=> 10
zubr >>> ((gattr third-attr my-record) 40)
=> 53
zubr >>> (subrecord my-record2 my-record
           (fourt-attr 55))
=> None
zubr >>> (gattr first-attr my-record2)
=> 10
```

Stdlib sources
-------------------

There are some basic functions defined in stdlib/, which constitutes a
kind of standard library.


Python scripts
-------------------

Rather than modifying the underlying lisp language directly by adding
more functionality in the form of python functions, the function``(py-script "/path/to/script")``
allows you to load python functions from a python script file to use
in lisp (e.g., in the case where non-trivial I/O functionality is
needed, or functions from an existing python library). 

For example, here is a standard python script:


```python

## script located as some path 

def newfunction(x):
    """A example function to use in lisp

    :param x: some integer value x 
    :returns: a list of even integers in the range or zero to x
    """
    return [i for i in range(0,x+1) if x % 2 == 0]

```

This function can be used by doing the following:

```lisp

zubr >>> (py-script "path/to/script")
=> None
zubr >>> newfunction
=> <function newfunction at ...>
zubr >>> (newfunction 10)
=> [0, 2, 4, 6, 8, 10]

```
