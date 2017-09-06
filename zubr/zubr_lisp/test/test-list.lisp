
;; list positions 

(test-assert-equal (first '(1 2 3)) 1)
(test-assert-equal (second '(1 2 3)) 2)
(test-assert-equal (third '(1 2 3)) 3)
(test-assert-equal (last '(1 2 3)) 3)
(test-assert-equal (last '(1 2 3)) (third '(1 2 3)))

;; list flatten

(test-assert-equal (flatten '((1 2)(3 4))) '(1 2 3 4))
(test-assert-equal (flatten '(())) '())
(test-assert-equal (flatten '()) '())

