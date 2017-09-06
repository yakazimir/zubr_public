;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : dict.lisp                              ;;
;; author      : Kyle Richardson                        ;;
;; description : defines dictionary functions/utilities ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun make-dict (list1 list2)
  "a function for making a dictionary from two lists"
  (dictionary (zip list1 list2)))

(defun filter-dict (dict-list key-list)
  "Filter a dictionary to include only key-vals from the keylist"
  (filter
   (lambda (d) (not (is-empty-container? d)))
   (map (lambda (x)
          (make-dict key-list
                     (map (lambda (y) (get-val x y)) key-list)))
        dict-list)))

(defun if-match-merge (dict1 dict2)
  "Determines if all overlapping matching keys overlap in values"
  (let ((overlap (filter
                  (lambda (x) (not (is-none? x)))
                  (map (lambda (z)
                         (if (is-none? (get-val dict1 z None))
                             None
                           (if (= (get-val dict2 z) (get-val dict1 z))
                               True
                             False)))
                       (keys dict2)))))
    ;overlap))
    (if (and (not (is-empty-container? overlap))
             (not (contains False overlap)))
        (update dict1 dict2)
      None)))

; this assumes (vals x) and (keys x) always returns same order

(defun update-key (dict oname nname)
  "Change the name of a key"
  (make-dict (map
              (lambda (x) (if (= x oname)
                              nname
                            x))
              (keys dict))
             (vals dict)))

