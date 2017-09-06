(define ex1 (make-dict '() '()))

(define ex2 (make-dict '(1 2) '(3 4)))

(define ex3
  (list (make-dict '(1 2) '(3 4))
        (make-dict '(1 2) '(5 6))))

(test-assert-true (is-empty-container? ex1))
(test-assert-false (is-empty-container? ex2))
(test-assert-true (not (is-empty-container? ex2)))
(test-assert-equal (make-dict '() '())
                   (make-dict '() '()))

(test-assert-equal (if-match-merge ex1 ex1) None)
(test-assert-equal (if-match-merge ex2 ex2) ex2)


(test-assert-equal (if-match-merge
                    (make-dict '(1 2) '(3 4))
                    (make-dict '(1 2 3) '(3 4 5)))
                   (make-dict '(1 2 3) '(3 4 5)))


(test-assert-equal (if-match-merge
                    (make-dict '(1 2 3) '(3 4 5))
                    (make-dict '(1 2 3 5) '(3 4 5 7)))
                   (make-dict '(1 2 3 5) '(3 4 5 7)))


(test-assert-none (if-match-merge
                    (make-dict '(1 2 3) '(3 4 5))
                    (make-dict '(1 2 3 5) '(3 7 5 7))))


(test-assert-equal
 (update-key
  (make-dict '(1 2 3) '(3 4 5))
  1 7)
 (make-dict '(7 2 3) '(3 4 5)))

 
 
