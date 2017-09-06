
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                                                               ;;
;; file        : database.lisp                                   ;;
;; author      : Kyle Richardson                                 ;;
;; description :  lisp implementation of relation database model ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Two core types, Relation,Database, and DatabaseContainer

(global-record
 (new-record Relation
             (name None) 
             (attributes None)
             (rows None)))

(global-record
 (new-record Database
             (name None)
             (relation-list None)
             (relations None)))

(global-record
 (new-record DatabaseContainer
             (name None)
             (database-list None)
             (databases None)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; macros for building relations and databases

(defmacro make-relation (rname attrs rlist) 
    "make a relation or table from a list of attributes and list of values"
  `(subrecord ,rname Relation
              (name ,(string rname))
              (attributes ,attrs)
              (rows
               (map (lambda (x) (make-dict ,attrs x))
                    ,rlist))))

(defmacro make-database (dname rels)
  "create a database from a list of relations"
  `(subrecord ,dname Database
              (name ,(string dname))
              (relation-list
               (make-dict
                (map (lambda (x) (:name x))
                     ,rels)
                (range (length ,rels))))
              (relations ,rels)))

(defmacro make-database-container (cname dbases)
  "Create a database container type"
  `(subrecord ,cname DatabaseContainer
              (name ,(string cname))
              (database-list
               (make-dict
                (map (lambda (x) (:name x))
                     ,dbases)
                (range (length ,dbases))))
              (databases ,dbases)))


(defmacro new-relation (name attrs rlist)
  "Allows you to define a new relation variable, where the name has the same as the relation name"
  `(define ,name
     (make-relation ,name ,attrs ,rlist)))

(defmacro generate-relation (rname attrs rlist)
  "Create a new relation variable where everyhing is already computed"
  `(subrecord ,rname Relation
              (name ,rname) ;,(string rname))
              (attributes ,attrs)
              (rows ,rlist)))
    

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Empty relation

(global-record
 (make-relation EMPTY-REL '() `()))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; comparison functions

(defun attributes-match (rel1 rel2)
  "Check if two relations have the same attributes"
  (if (= (set-obj (:attributes rel1))
         (set-obj (:attributes rel2)))
      True
    False))

(defun attribute-overlap (rel1 rel2)
  "Checks if list of attributes overlapp between rel1 and rel2"
  (if (> (length
          (filter (lambda (x)
                    (contains x (:attributes rel2)))
                  (:attributes rel1)))
         0)
      True
    False))

(defun row-len-eq (rel1 rel2)
  "Check if the number of tuples match"
  (if (= (len (:rows rel1)) (len (:rows rel2)))
      True
    False))

(defun row-size (rel)
  "The number of rows"
  (length (:rows rel)))

(defun check-in-row (rel)
  "Checks if value is in another row"
  (lambda (x) (contains x (:rows rel))))

(defun check-not-in-row (rel)
  "Checks if value is not in another row"
  (lambda (x) (not (contains x (:rows rel)))))

(defun tuple-values-in (rel1 rel2)
  "Checks if "
  (map-cond (check-in-row rel2)
            (:rows rel1)))

(defun is-empty-rel (relation)
  "Determines if a relation is empty (has no row entries or tuples)" 
  (if (not (:rows relation))
      True
    False))

(defun MINUS (rel1 rel2)
  "The relation result from taking away overlapping values in rel2 and rel1"
  (if (attributes-match rel1 rel2)
      (generate-relation (concat (:name rel1) "-minus")
                         (:attributes rel1)
                         (map-cond (check-not-in-row rel2)
                                   (:rows rel1)))
    rel1))

(defun is-equal-rel (rel1 rel2)
  "Determines if two relations are equal"
  (if (attributes-match rel1 rel2) ;; make to make a set
      (if (row-len-eq rel1 rel2)
          (if (is-empty-rel (MINUS rel1 rel2))
              True
            False)
        False)
    False))

(defun not-equal-rel (rel1 rel2)
  "Determines if two rels are not equal"
  (not (is-equal-rel rel1 rel2)))

(defun is-subset-rel (rel1 rel2)
  "Determines if rel1 is a subset of rel2"
  (if (= (len (tuple-values-in rel1 rel2))
         (len (:rows rel1)))
      True
    False))

(defun is-proper-subset-rel (rel1 rel2)
  "Determines if rel2 is a proper subset of rel2"
  (if (is-subset-rel rel1 rel2)
      (if (not-equal-rel rel1 rel2)
          True
        False)
    False))


;; ;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; ;; ;; Implementation of the relational algebra


;; ;; unary operators

(defun RESTRICT (relation condition)
  "Create a new table restricted to a condition"
  (generate-relation (concat (:name relation) "*")
                     (:attributes relation)
                     (map-cond condition
                               (:rows relation))))

(defun PROJECT (relation key-list)
  "Create new table/relation given with only certain attributes"
  (generate-relation (concat (:name relation) "-")
                     (filter (lambda (a)
                               (contains a key-list))
                             (:attributes relation))
                     (filter-dict (:rows relation)
                                  key-list)))

;; binary operators

; pure composition 

(defun TIMES (rel1 rel2)
  "Compose two relations (assumes relations are disjoint)"
  (generate-relation (concat
                      (:name rel1) "+" (:name rel2))
                     (+ (:attributes rel1)
                        (:attributes rel2))
                     (map-flatten
                      (lambda (x)
                             (map (lambda (y)
                                    (update x y))
                                  (:rows rel2)))
                           (:rows rel1))))

(defun NATURAL-JOIN (rel1 rel2)
  "Joins two items with overlapping (some or all) values"
  (generate-relation (concat
                      (:name rel1) "&" (:name rel2))
                     (+ (:attributes rel1)
                        (filter (lambda (a)
                                  (not (contains a (:attributes rel1))))
                                (:attributes rel2)))
                     (map-flatten
                      (lambda (x)
                        (map-flatten
                         (lambda (y)
                           (filter None 
                                   (list (if-match-merge x y))))
                         (:rows rel2)))
                      (:rows rel1))))

; should condition on attribute overlap/mismatch 

(defun SEMI-JOIN (rel1 rel2)
  "Joins two items them projects the resulting relation with itself"
  (PROJECT (NATURAL-JOIN rel1 rel2) (:attributes rel1)))


(defun INTERSECT (rel1 rel2)
  "Join two relations that have the same attributes"
  (generate-relation (concat
                      (:name rel1) "&*" (:name rel2))
                     (:attributes rel1)
                     (map-cond (lambda (x)
                                 (contains x (:rows rel2)))
                               (:rows rel1))))
                                          
                     
;; joint function will either do composition or semi-join (?) depending
;; on whether the attributes overlap

(defun AND (rel1 rel2)
  "Join two relations rel1 and rel2"
  (if (attributes-match rel1 rel2)
      (INTERSECT rel1 rel2)
    (if (attribute-overlap rel1 rel2)
        (NATURAL-JOIN rel1 rel2)
      (TIMES rel1 rel2))))

;; ;;; rename operator

(defun RENAME (rel oname nname)
  "Rename an attributes"
  (generate-relation (concat (:name rel) "+<rename>")
                     (map (lambda (x)
                            (if (= x oname)
                                nname
                              x))
                          (:attributes rel))
                     (map (lambda (r)
                            (update-key r oname nname))
                          (:rows rel))))

;; ;; aggregate relations

(defun COUNT (rel)
  "Returns the number of rows"
  (length (:rows rel)))

(defun find-value-list (rel val_or_function)
  "Returns list of values to apply an aggregrate function on"
  (let ((func (if (callable? val_or_function)
                 val_or_function
               (lambda (x) (= x val_or_function)))))
    (map-flatten
     (lambda (tuple)
       (filter None
               (map (lambda (tval)
                      (if (func tval)
                          (get-val tuple tval)
                        None))
                    tuple)))
     (:rows rel))))


(defun aggregrate-fun (rel func val_or_function)
  "Generation function for aggregate function"
  (func (find-value-list rel val_or_function)))

(defun SUM (rel attribute)
  "Sums a list of attribute values"
  (aggregrate-fun rel sum attribute))

(defun MAX (rel attribute)
  "Return the maximum value in a list"
  (aggregrate-fun rel max attribute))

(defun MIN (rel attribute)
  "Returns the minimum value in an attribute list"
  (aggregrate-fun rel min attribute))

(defun AVERAGE (rel attribute)
  "Averages a list of attribute values"
  (let ((val-list (find-value-list rel attribute)))
    (let ((size (length val-list))
          (sum-val (sum val-list)))
      (if (= sum-val 0)
          0
        (/ (float sum-val)
           (float size))))))

;;; generalized quantifiers

;; possibility combinations

;; GQ (relation relation) ;; all (files) (modified)
                          ;; all (modified) (files) 


;; GQ (function relation) ;; several (python) (functions)
                          ;; a few (large) (files)

;; not allowed: GQ (relation function)
;;              GQ (function function)


(defun gq-set (rl1 rl2)
  "Returns the resulting item to compare using generalized quantifers"
  (if (and (not (is-record? rl1))
           (not (is-record? rl2)))
      (list EMPTY-REL False)
    (if (not (is-record? rl2))
        (list EMPTY-REL False)
      (if (not (is-record? rl1))
          (list (RESTRICT rl2 rl1) False)
        (if (attribute-overlap rl2 rl2)
            (list (AND rl1 rl2) True)
          (list EMPTY-REL True))))))

;; (defun GQ (rl1 rl2 fn1 fn2)
;;   "General implementation of a generalized quantifier"
;;   (let ((result-list (gq-set rl1 rl2)))
;;     (if (is-true? (last result-list))
;;         (fn


(defun ALL (rl1 rl2)
  "Implementation of the ALL quantifier"
  (let ((result-list (gq-set rl1 rl2)))
    (let ((result (first result-list))
          (is-rel (last result-list)))
      (if (istrue? is-rel) ;; is a relation
          (if (= (row-size rl1)
                 (row-size result))
              result
            EMPTY-REL)
        result))))

(defun SOME (rl1 rl2)
  "Implementation of the SOME quantifier"
  (let ((result-list (gq-set rl1 rl2)))
    (let ((result (first result-list))
          (is-rel (last result-list)))
      (if (istrue? is-rel)
          (if (>= (row-size result) 1)
              result
            EMPTY-REL)
        result))))


(defun NO (rl1 rl2)
  "Implementation of the NO quntifier"
   (let ((result-list (gq-set rl1 rl2)))
    (let ((result (first result-list))
          (is-rel (last result-list)))
      (if (is-empty-rel result)
          rl2
        EMPTY-REL))))
  
  
;;; utility functions

(defun relation-str (relation)
  "returns a string representation of a relation"
  (let ((attribute-row
         (join
          (map (lambda (x)
                 (format-string "|{0: <20}" x))
               (:attributes relation)) " "))
        (divider ;; this is pretty ghetto
         (join 
          (map (lambda (x)
                 (join (map (lambda (y) "-") (range 20)) "" ))
               (:attributes relation)) ""))                 
        (value-rows
         (join
          (map (lambda (x)
                 (newline
                  (join (map (lambda (y)
                              (format-string " {0: <20}" (get-val x y)))
                            (:attributes relation)) " ")))
               (:rows relation)) "")))
    (+ (newline attribute-row)
       (+ (newline divider) value-rows))))

;; (defun database-str (database)
;;   "A string representation of a database"
;;   "Database")

;;; database operations

(defun print-rel (relation)
  (display (relation-str relation)))


;; querying a database

;; (query (db query)



;;; infers the type of operators


