
(load "database.lisp")

;; Relation record 

(test-assert-equal (:name Relation) None)
(test-assert-equal (:attributes Relation) None)
(test-assert-equal (:rows Relation) None)

;; Database record

(test-assert-equal (:name Database) None)
(test-assert-equal (:relation-list Database) None)
(test-assert-equal (:relations Database) None)

;; Database container

(test-assert-equal (:name DatabaseContainer) None)
(test-assert-equal (:database-list DatabaseContainer) None)
(test-assert-equal (:databases DatabaseContainer) None)

;; make relation

(define my-relation
  (make-relation MyRelation
                 '(attr1 attr2 attr3)
                 `((attr1-val attr2-val attr3-val)
                   (attr21-val attr22-val attr23-val))))

(define my-relation9
  (make-relation My9
                 '(v1 v2)
                 `((value1 value2))))


(define my-relation2
  (make-relation MyRelation2
                 '(attr1 attr2)
                 `((attr1-val attr2-val))))

(define my-relation7
  (make-relation MyRelation2
                 '(attr2 attr1)
                 `((attr2-val attr1-val))))

(define my-relation3
  (make-relation MyRelation3
                 '(attr1 attr2 attr3)
                 `((attr1-val attr2-val attr3-val))))

;; attribute overlap
(test-assert-true (attribute-overlap my-relation7 my-relation3))
(test-assert-true (attribute-overlap my-relation3 my-relation7))
(test-assert-true (attribute-overlap my-relation my-relation3))
(test-assert-false (attribute-overlap my-relation my-relation9))
(test-assert-false (attribute-overlap my-relation9 my-relation))
(test-assert-false (attribute-overlap my-relation9 my-relation7))

(test-assert-true (attributes-match my-relation2 my-relation7))
(test-assert-true (is-equal-rel my-relation2 my-relation7))

(test-assert-equal (:attributes my-relation) '(attr1 attr2 attr3))
(test-assert-not-equal (:attributes my-relation) '(attr2 attr3 attr1))
(test-assert-equal (length (:attributes my-relation)) 3)
(test-assert-equal (length (:rows my-relation)) 2)

;; comparisons

(test-assert-false (is-empty-rel my-relation))
(test-assert-true (not (is-empty-rel my-relation)))
(test-assert-true (is-equal-rel my-relation my-relation))
(test-assert-false (not-equal-rel my-relation my-relation))
(test-assert-true (is-subset-rel my-relation my-relation))
(test-assert-false (is-proper-subset-rel my-relation my-relation))

;; comparison with different

;; equivlanece 
(test-assert-true (not-equal-rel my-relation my-relation2))
(test-assert-true (not-equal-rel my-relation2 my-relation))

;;
(test-assert-true (is-subset-rel my-relation3 my-relation))
(test-assert-true (is-proper-subset-rel my-relation3 my-relation))
(test-assert-false (is-subset-rel my-relation my-relation3))

;; tuple-values in

(test-assert-equal (tuple-values-in my-relation my-relation2) '())
(test-assert-equal (tuple-values-in my-relation2 my-relation) '())

(test-assert-equal
 (tuple-values-in my-relation2 my-relation2) (:rows my-relation2))

(test-assert-equal
 (tuple-values-in my-relation my-relation2) '())

(test-assert-equal
 (tuple-values-in my-relation my-relation3)
 (list (make-dict '(attr1 attr2 attr3)
                  '(attr1-val attr2-val attr3-val))))

;; minus relation tests

(test-assert-equal
 (:rows (MINUS my-relation my-relation3))
 (list (make-dict '(attr1 attr2 attr3)
                  '(attr21-val attr22-val attr23-val))))


(test-assert-equal
 (:rows (MINUS my-relation my-relation)) '())

(test-assert-equal
 (:rows (MINUS my-relation my-relation2))
 (:rows my-relation))

(test-assert-equal
 (:rows (MINUS my-relation2 my-relation))
 (:rows my-relation2))

;;
;;;;;; testing the relational algebra stuff
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;from the D tutorial

(new-relation IS_CALLED
              '(StudentId Name)
              `((S1 Anne)
                (S2 Boris)
                (S3 Cindy)
                (S4 Devinder)
                (S5 Boris)))

(new-relation IS_ENROLLED_ON
              '(StudentId CourseId)
              `((S1 C1)
                (S1 C2)
                (S2 C1)
                (S3 C3)
                (S4 C1)))

(new-relation BORIS
              '(StudentId Name)
              `((S2 Boris)
                (S5 Boris)))

(new-relation STUDENT_NAME 
              '(Name)
              `((Anne)
                (Boris)
                (Cindy)
                (Devinder)
                (Boris)))


(test-assert-equal (:attributes IS_CALLED) '(StudentId Name))
(test-assert-equal (:attributes IS_ENROLLED_ON) '(StudentId CourseId))

;; restrictions

(define boris-restriction
  (RESTRICT IS_CALLED
                (lambda (x) (= (:Name x) "Boris"))))

(test-assert-equal (:rows boris-restriction) (:rows BORIS))

(test-assert-equal (:rows (RESTRICT IS_CALLED
                                        (lambda (x) x)))
                   (:rows IS_CALLED))

; projection

(test-assert-equal (:rows (PROJECT IS_CALLED '("Name")))
                   (:rows STUDENT_NAME)) 

(test-assert-equal (:rows (PROJECT IS_CALLED '(StudentId Name)))
                   (:rows IS_CALLED))


(test-assert-equal (:rows (PROJECT IS_CALLED '(Name StudentId)))
                   (:rows IS_CALLED))

(test-assert-equal (:attributes (PROJECT IS_CALLED '(Name StudentId)))
                   '(StudentId Name))



(test-assert-equal (:rows (PROJECT IS_CALLED '())) '())

;; composition

(global-record
 (make-relation student-gradyear
                '(student year)
                `((john 2010)
                  (same 2000)
                  (joan 2012))))

(global-record 
 (make-relation student-school
                '(student school)
                `((john nyu)
                  (same stanford))))

(global-record 
 (make-relation random-rel
                '(a b)
                `((john 2010)
                  (sarah 2007))))


(define random-compose (TIMES student-school random-rel))

(test-assert-equal (:attributes random-compose)
                   '(student school a b)
                   )

(test-assert-equal (length (:rows random-compose)) 4)

(test-assert-equal (:rows random-compose)
                   (list
                    (make-dict '(student school a b)
                               '(john nyu john 2010))
                    (make-dict '(student school a b)
                               '(john nyu sarah 2007))
                    (make-dict '(student school a b)
                               '(same stanford john 2010))
                    (make-dict '(student school a b)
                               '(same stanford sarah 2007))
                    ))

;; (test-assert-equal (:name random-compose)
;;                    (concat (:name student-school) "+" (:name random-rel)))

;; natural join

(define join-overlap-ex (NATURAL-JOIN student-gradyear student-school))

(test-assert-equal (length (:rows join-overlap-ex)) 2)

(test-assert-equal (:attributes join-overlap-ex) '(student year school))

(test-assert-equal (:rows join-overlap-ex)
                   (list
                    (make-dict '(student year school)
                               '(john 2010 nyu))
                    (make-dict '(student year school)
                               '(same 2000 stanford))
                    ))


(test-assert-equal (:rows (INTERSECT student-gradyear student-gradyear))
                   (:rows student-gradyear)
                   )

(test-assert-equal (:attributes (INTERSECT student-gradyear student-gradyear))
                   (:attributes student-gradyear)
                   )

;; join operator functino

(test-assert-equal (:rows (AND student-gradyear student-school))
                   (:rows join-overlap-ex)
                   )

(test-assert-equal (:rows (AND student-school random-rel))
                   (:rows (TIMES student-school random-rel))
                   )

(test-assert-equal (:rows (TIMES student-school random-rel))
                   (:rows (AND student-school random-rel))
                   )

(test-assert-equal (:rows (AND student-school student-gradyear))
                   (:rows (AND student-gradyear student-school))
                   )

;; semi-join

(test-assert-equal (:rows (SEMI-JOIN student-school student-school))
                   (:rows student-school))

;; rename

(test-assert-equal (:attributes (RENAME student-gradyear "year" "new"))
                   '(student new))


;; important, join with more than one overlapping attributes

(global-record
 (make-relation relation1
                '(attr1 attr2 attr3)
                `((a1 a2 a3)
                  (aa1 aa2 aa3))))

(global-record
 (make-relation relation2
                '(attr1 attr2 attr3)
                `((a1 a7 a3)
                  (aaa1 aa2 aa3))))

(test-assert-equal (:rows (AND relation1 relation2))
                   '())

;; aggregate relations

(test-assert-equal (COUNT student-school) 2)
(test-assert-equal (COUNT my-relation) 2)
(test-assert-equal (COUNT (AND student-school my-relation)) 4)

(global-record
 (make-relation newentry
                '(attr1 attr2 attr3)
                '((1 2 3)
                  (1 2 3)
                  (1 2 3)
                  (1 2 10))
                ))

(test-assert-equal (MIN newentry "attr3") 3)
(test-assert-equal (MAX newentry "attr3") 10)
(test-assert-equal (SUM newentry "attr3") 19)

; might want to return a float 
(test-assert-equal (AVERAGE newentry "attr3") 4.75)

;; empty lists


