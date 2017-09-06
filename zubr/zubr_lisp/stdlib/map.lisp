;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file        : map.lisp                               ;;
;; author      : Kyle Richardson                        ;;
;; description : defines map/iteration functions        ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; since there I haven't implemented iteration, for now I'm using
;; maps for everything (sometimes in a rather rediculous way)

(defun map-iter (cont1 cont2 fn)
  "iterate for "
  (flatten
   (map
    (lambda (x)
      (map 
       (lambda (y)
         (fn x y))
       cont2))
    cont1)))


(defun map-filter (function list filter-value)
  "Apply a map then a filter on map output"
  (filter filter-value
          (map function list)))

(defun make-cond (function)
  "Takes a function and puts it into a conditional"
  (lambda (x) (if (function x) x None)))

(defun map-cond (condition list)
  "Maps a list with a condition and filters those that map to None "
  (filter None
          (map (make-cond condition) list)))

(defun map-flatten (map-fun list)
  "Applies a flatten operation to a map"
  (flatten (map map-fun list)))
