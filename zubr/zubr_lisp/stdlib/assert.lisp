;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : assert.lisp                            ;;
;; author      : Kyle Richardson                        ;;
;; description : defines assertion functions            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defun assert-equal (lispexpr target)
  "Returns true is equal and false otherwise"
  (if (= (begin lispexpr) target)
      True
    False))

(defun assert-not-equal (lispexpr target)
  "Return true when lispexpr is not equal to target"
  (if (assert-equal lispexpr target)
      False
    True))

(defun assert-true (lispexpr)
  "Assert that a lisp expression is True"
  (if (assert-equal lispexpr True)
      True
    False))

(defun assert-false (lispexpr)
  "Assert that a lisp expression is False"
  (if (assert-equal lispexpr False)
      True
    False))

(defun assert-none (lispexpr)
  "Assert that expression maps to nil"
  (if (assert-equal lispexpr None)
      True
    False))

(defun assert-complete (lispexpr)
  "Assert that something will complete"
  listexpr)
