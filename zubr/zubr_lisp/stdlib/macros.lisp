;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : macros.lisp                            ;;
;; author      : Kyle Richardson                        ;;
;; description : basic macro definitions in lisp stdlin ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define-macro and
  ; example and macro from norvig
  (lambda args
    (if null? args)
    True
    (if (= (length args) 1)
        (car args)
      `(if ,(car args)
           (and ,@(cdr args))
         False))))

(define-macro unless
  ; example from norvig 
  (lambda args `(if (not ,(car args))
                    (begin ,@(cdr args)))))

(define-macro when
  ;; when condition
  (lambda args
    (let ((condition (nth args 0))
	(exp (cdr args)))
      `(if ,condition
	   (begin ,@exp)))))
