
;; (define-macro lambda-recur
;;   ; recursive building lambda abstractors 
;;   (lambda args
;;     (let ((input-arguments (first args))
;;           (body (last args)))
;;       (if (is-empty-container? input-arguments)
;;           body
;;         `(lambda (,(first input-arguments))
;;            (lambda-recur ,(cdr input-arguments) ,body))))))


(defmacro lambda-recur args
  "Distributes lambdas over all arguments"
  (let ((input-arguments (first args))
        (body (last args)))
    (if (is-empty-container? input-arguments)
        body
      `(lambda (,(first input-arguments))
         (lambda-recur ,(cdr input-arguments) ,body)))))

(defmacro defcurry args
  "Auto curries a defined input function"
  (let ((arglen (length args)))
    (let ((name (first args))
          (aargs (second args))
          (body (if (= arglen 4)
                    (nth args 3)
                  (third args))))
      `(define ,name (lambda-recur ,aargs ,body)))))
