;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : core.lisp                              ;;
;; author      : Kyle Richardson                        ;;
;; description : defines The most basic lisp functions  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(define-macro defn 
  (lambda args
    (define is-four (= (length args) 4))
    (let ((name (nth args 0))
	  (args (nth args 1))
	  (body (if (= (length args) 4)
	  	    (nth args 3)
                  (nth args 2))))
      `(define ,name (lambda ,args ,body)))))


;; defining functions (as above)

(define-macro defun
  (lambda args
    (let ((name (nth args 0))
	  (args (nth args 1))
	  (body (if (= (length args) 4)
	  	    (nth args 3)
                  (nth args 2))))
      `(define ,name (lambda ,args ,body)))))

;; defining macros

(define-macro defmacro
  (lambda args
    (let ((name (nth args 0))
          (args (nth args 1))
          (body (if (= (length args) 4)
                    (nth args 3)
                  (nth args 2))))
      `(define-macro ,name (lambda ,args ,body)))))


;; general container functions

(defun is-empty-container? (container)
  "Deteremines if a container object is empty"
  (if (= (len container) 0)
      True
    False))


;; auto-curry

;;(define-macro defcurry
  ;; define a curried function
  ;;(let 
  ;; (let ((name (nth args 0))
  ;;       (args (nth args 1))
  ;;       (arglen (length args))
  ;;       (body (if (= (length args) 4)
  ;;                 (nth args 3)
  ;;               (nth args 2))))
;;   True))

  
