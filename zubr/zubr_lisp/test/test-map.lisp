
;; example functions 

(define fun1 (lambda (x) (if (> x 1) x None)))
(define fun2 (lambda (y) (not (= y None))))

;; map condition

;; map filters 

(test-assert-equal (map-filter fun1 '(1 2 3 4) None) '(2 3 4))
(test-assert-equal (map-filter fun1 '(1 2 3 4) fun2) '(2 3 4))
(test-assert-equal (map-cond fun1 '(20 30 4 0 -1)) '(20 30 4))
