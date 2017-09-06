
(load "database.lisp")

;; example tables


(global-record 
 (make-relation student-school
                '(student school)
                `((john nyu)
                  (same stanford)
                  (mary mit)
                  (bill princeton)
                  (tom utexas)
                  (steve berkeley)
                  (francis rit)
                  (joan upenn)
                  (sarah bu))))

(global-record
 (make-relation student-gradyear
                '(student year)
                `((john 2010)
                  (same 2000)
                  (mary 1998)
                  (bill 1993)
                  (tom 2015)
                  (steve 2000)
                  (francis 2000)
                  (joan 2012)
                  (sarah 2007))))

(global-record
 (make-relation random-rel
                '(a b)
                `((john 2010)
                  (same 2000)
                  (mary 1998)
                  (bill 1993)
                  (tom 2015)
                  (steve 2000)
                  (francis 2000)
                  (joan 2012)
                  (sarah 2007))))

(make-database my-database
               (list student-school
                     student-gradyear))

;; new defintion of make-database


(define hi-people (TIMES student-gradyear random-rel))

;(define hi-people (join-overlap student-gradyear student-school))

;(define my-condition (lambda (x) (> (:year x) 2000)))
;(define newrel (restrict-rel student-gradyear my-condition))
;(print (string (:rows (restrict-rel student-gradyear my-condition))))
;(print (string (:rows (projec-rel student-gradyear '("year")))))

;; (define my-restrict (restrict-rel student-gradyear my-condition))
;; (define year-only (project-rel student-gradyear '("year")))
