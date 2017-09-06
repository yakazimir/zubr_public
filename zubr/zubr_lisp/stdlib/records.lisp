;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : recrods.lisp                           ;;
;; author      : Kyle Richardson                        ;;
;; description : defines record abstraction             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;
;; zubr lisp has a ``record`` abstraction, which is a simple type
;; of (method-less) class similar clojure's record abstraction.
;; The macros below are simple ways for building these records,
;; either from a simple object, or from another record
;;

(defmacro new-record args
  "macro for building a ``simple`` record (i.e., inherits from object)"
  `(define-record ,(first args) object (attr ,@(cdr args))))

(defmacro subrecord args
  "macro for building a ``subrecord``, or inherited record (from some other than object)"
  `(define-record ,(first args) ,(nth args 1) (attr ,@(cdr args))))

(defmacro gattr args
  "simple macro for finding attributes without having to string quote attr names"
  `(r-attr ,(string (first args)) ,(nth args 1)))


(defmacro sattr args
  "simple macro for setting attributes without string quoting"
  `(s-attr ,(string (first args)) ,(nth args 1) ,(last args)))
