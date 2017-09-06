;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : bool.lisp                              ;;
;; author      : Kyle Richardson                        ;;
;; description : defines boolean functions/conditions   ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defun istrue? (x)
  "Deteremines if something is true"
  (eq? x True))

(defun is-none? (x)
  "Determines if something is nil"
  (= x None))
