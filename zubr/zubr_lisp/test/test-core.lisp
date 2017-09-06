
;; PYTHON IMPLEMENTATION

(test-assert-false (isinstance '() float))
(test-assert-true (isinstance 3.4 float))
(test-assert-false (isinstance 3.4 int))
(test-assert-true (isinstance 3 int))

;;; LIST FUNCTIONS

(test-assert-equal (list 1 2 3) '(1 2 3))
(test-assert-equal (list '(1 2 3)) '((1 2 3)))
(test-assert-equal (list) '())

(test-assert-equal (car '(())) '())
(test-assert-equal (car '(1 2 3)) 1)
(test-assert-equal (car '(2 3)) 2)
(test-assert-equal (cdr '(1 2 3)) '(2 3))
(test-assert-equal (car (cdr '(1 2 3))) 2)
(test-assert-equal (cdr '()) '())
(test-assert-equal (cdr '(()())) '(()))
(test-assert-true  (null? '()))
(test-assert-false  (null? '(1)))
(test-assert-false  (null? '(())))
(test-assert-true  (list? '()))
(test-assert-true  (list? '(1 2 3)))
(test-assert-false (list? 4))
(test-assert-false (list? "4"))
(test-assert-equal (length '()) 0)
(test-assert-equal (length '(1)) 1)
(test-assert-equal (length '(1)) (len '(1)))
(test-assert-equal (length '()) (len '()))

;; arithmetic

(test-assert-equal (+ 10 10) 20)
(test-assert-equal (- 10 10) 0)

;; comaprison

(test-assert-false (> 10 20))
(test-assert-true (< 10 20))
(test-assert-true (>= 10 10))
(test-assert-true (<= 10 100))



