;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : test.lisp                              ;;
;; author      : Kyle Richardson                        ;;
;; description : defines unit testing functions         ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(global-record
 (new-record TestResult
             (msg None)
             (result None)))

;; pass record 

(global-record
 (subrecord Pass
            TestResult
            (result True)))


;; fail record 

(global-record
 (subrecord Fail
            TestResult
            (result False)))


;; test-assertions : returns either pass or fail record 

(defun test-assert-equal (lispexpr target)
  "Returns true if assertion holds, otherwise returns mismatched values"
  (if (assert-equal lispexpr target)
      Pass
    (subrecord EqFail
               Fail
               (msg (format-string "Assert error! Expected {0}, received {1}"
                                   (string target)
                                   (string lispexpr))))))


(defun test-assert-not-equal (lispexpr target)
  "Returns true is negation of assertion is true"
  (if (assert-not-equal lispexpr target)
      Pass
    (subrecord NEFail
               Fail
               (msg (format-string "Assert error! Both are equal to {0}"
                                   (string target))))))


(defun test-assert-true (lispexpr)
  "Returns true if lispexpression returns True"
  (if (assert-true lispexpr)
      Pass
    (subrecord TrueFail
               Fail
               (msg (format-string "Assert error! Returns False")))))


(defun test-assert-false (lispexpr)
  "Returns true if lispexpression returns True"
  (if (assert-false lispexpr)
      Pass
    (subrecord FalseFail
               Fail
               (msg (format-string "Assert error! Returns True")))))


(defun test-assert-none (lispexpr)
  "Reeturn true is lisp expression evaluate to None"
  (if (assert-none lispexpr)
      Pass
    (subrecord NoneFail
               Fail
               (msg (format-string "Assert error! Expression is not nil")))))
