;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : list.lisp                              ;;
;; author      : Kyle Richardson                        ;;
;; description : defines python list functions/utilities;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defun first (lst)
  "Returns the first item in a list"
  (nth lst 0))

(defun second (lst)
  "Returns the second item in a list"
  (nth lst 1))

(defun third (lst)
  "Returns the third item in a list"
  (nth lst 2))

(defun last (lst)
  "Returns the final item in a list"
  (nth lst -1))

;; might have efficiency issues

(defun flatten (list-of-lists)
  "flatten a list of lists"
  (sum list-of-lists '()))

