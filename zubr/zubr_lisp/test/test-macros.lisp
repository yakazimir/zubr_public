
;;;;;;;;;;;;;;;;;;;;;;
;;; conjunction 

(test-assert-false (and True False))
(test-assert-true (and True True))
(test-assert-false (and False False))

;; single items

(test-assert-true (and True))
(test-assert-false (and False))
(test-assert-true (not (and False)))
(test-assert-false (not (and True)))
(test-assert-true (not (not (and True))))

