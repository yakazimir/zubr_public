;;;;;;;;;;;;;;;;;;;;;;
;; core.lisp tests  ;;
;;;;;;;;;;;;;;;;;;;;;;

;; list operations

(test-assert-equal (first '(1 2 3 4)) 1)
(test-assert-equal (second '(1 2 3 4)) 2)
(test-assert-equal (third '(1 2 3 4)) 3)
(test-assert-equal (last '(1 2 3 4)) 4)

;; make a dictionary

(test-assert-equal (make-dict '(1) '(1))
                   (dictionary (zip '(1) '(1))))

;; is-true

(test-assert-true (istrue? (= 10 10)))

;;;;;;;;;;;;;;;;;;;;;;;
;; macros.lisp tests ;;
;;;;;;;;;;;;;;;;;;;;;;;


;; conjunction 

(test-assert-true (and True True))
(test-assert-false (and False True))
(test-assert-true (and (= 1 1) (not (= 10 11))))
(test-assert-false (and (= 1 1) (= 10 11)))

;; flatten

(test-assert-equal (flatten '((1 2 3) (4 5 6)))
                   '(1 2 3 4 5 6))

