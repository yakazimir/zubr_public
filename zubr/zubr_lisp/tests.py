from zubr.zubr_lisp.Lang import query

test_cases = [
    ("(+ 2 2)", 4),
    ("(+ (* 2 100) (* 1 10))", 210),
    ("(if (> 6 5) (+ 1 1) (+ 2 2))", 2),
    ("(if (< 6 5) (+ 1 1) (+ 2 2))", 4),
    ("(define x 3)", None),
    ("x", 3),
    ("(+ x x)", 6),
    ("(begin (define x 1) (set! x (+ x 1)) (+ x 1))", 3),
    ("((lambda (x) (+ x x)) 5)", 10),
    ("(define twice (lambda (x) (* 2 x)))", None),
    ("(twice 5)", 10),
    ("(define compose (lambda (f g) (lambda (x) (f (g x)))))", None),
    ("((compose list twice) 5)", [10]),
    ("(define repeat (lambda (f) (compose f f)))", None),
    ("((repeat twice) 5)", 20), ("((repeat (repeat twice)) 5)", 80),
    ("(define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))", None),
    ("(fact 3)", 6),
    ("(fact 50)", 30414093201713378043612608166064768844377641568960512000000000000),
    ("(define abs (lambda (n) ((if (> n 0) + -) 0 n)))", None),
    ("(list (abs -3) (abs 0) (abs 3))", [3, 0, 3]),
    ("""(define combine (lambda (f)
    (lambda (x y)
      (if (null? x) (quote ())
          (f (list (car x) (car y))
             ((combine f) (cdr x) (cdr y)))))))""", None),
    ("(define zip (combine cons))", None),
    ("(zip (list 1 2 3 4) (list 5 6 7 8))", [[1, 5], [2, 6], [3, 7], [4, 8]]),
    ("(zip (list 1 2 3 4) (list 5 6 7 8))", [[1, 5], [2, 6], [3, 7], [4, 8]]),
    ("(define zip (combine cons))", None),
    ("(define (twice x) (* 2 x))", None),
    ("(twice 2)", 4),
    ("(define lyst (lambda items items))", None),
    ("(lyst 1 2 3 (+ 2 2))", [1,2,3,4]),
    ("(if 1 2)", 2),
    ("(if (= 3 4) 2)", None),
    ("(define ((account bal) amt) (set! bal (+ bal amt)) bal)", None),
    ("(define a1 (account 100))", None),
    ("(a1 0)", 100),
    ("(a1 10)", 110), ("(a1 10)", 120),
    ("""(define (newton guess function derivative epsilon)
    (define guess2 (- guess (divide (function guess) (derivative guess))))
    (if (< (abs (- guess guess2)) epsilon) guess2
    (newton guess2 function derivative epsilon)))""", None),
    ("""(define (square-root a)
    (newton 1 (lambda (x) (- (* x x) a)) (lambda (x) (* 2 x)) 1e-8))""", None),
    ("(> (square-root 200.) 14.14213)", True),
    ("(< (square-root 200.) 14.14215)", True),
    ("""(define (sum-squares-range start end)
         (define (sumsq-acc start end acc)
            (if (> start end) acc (sumsq-acc (+ start 1) end (+ (* start start) acc))))
            (sumsq-acc start end 0))""", None),
    ("(sum-squares-range 1 3000)", 9004500500),
    ("(let ((a 1) (b 2)) (+ a b))", 3),
    ("(define-macro unless (lambda args `(if (not ,(car args)) (begin ,@(cdr args))))) ; test `", None),
    ("(unless (= 2 (+ 1 1)) (display 2) 3 4)", None),
    ("(quote x)", 'x'),
    ("(quote (1 2 three))", [1, 2, 'three']),
    ("'(one 2 3)", ['one', 2, 3]),
    ("(define L (list 1 2 3))", None),
    ("`(testing ,@L testing)", ['testing',1,2,3,'testing']),
    ("`(testing ,L testing)", ['testing',[1,2,3],'testing']),
    ## newer features
    ("(range 10)",range(10)),
    ("(nth (list 1 2 3 4) 0)",1),
    ("(set 1 2 3 4)",set([1,2,3,4])),
    ("(float 4)",4.0),
    ("""(int "4")""",4),
    ("(cond ((= 3 4) 4) (+ 4 5))",9),
    ("(cond ((= 3 3) 4) (+ 4 5))",4),
]

if __name__ == "__main__":
    for k,(x,expected) in enumerate(test_cases):
        try: 
            result = query(x)
            equal = result == expected
            print "test case %d: %s" % (k,equal)
            if not equal:
                print "\t result=%s" % (result)
                print "\t expected=%s" % (expected)
        except Exception as e:
            print "!! issue on %d !! \n\t %s" % (k,e)
