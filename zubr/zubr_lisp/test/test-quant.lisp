;; test of generalized quantifiers in the extended relational algebra


(new-relation rel1
              '(attr1 attr2 attr3)
              `((1 2 3)
                (3 4 5)
                (6 7 8)
                (9 10 11)))

(new-relation rel2
              '(attr1 attr2 attr3)
              `((1 2 3)
                (9 10 11)))

(new-relation rel3
              '(attr1 attr2 attr3)
              `((17 18 19)
                (14 13 20)))


;; negation 

(test-assert-equal (length (:rows (NO rel1 rel2)))
                   0)

(test-assert-equal (:rows (NO rel3 rel2))
                   (:rows rel2))

; remove all python files 

;; (ALL DELETED (RESTRICT FILES PYTHON))

;; (ALL PEOPLE HUMANS)



;; all python files

;; all

(test-assert-equal (:rows (ALL rel1 rel1))
                   (:rows rel1))

(test-assert-equal (:rows (ALL rel2 rel2))
                   (:rows rel2))
