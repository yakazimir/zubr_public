

(defun fun1 (x y) (+ x y))
(defcurry func (x y) (+ x y))

(test-assert-equal (fun1 10 20) ((func 10) 20))
(test-assert-equal (fun1 10 20) ((func 20) 10))

