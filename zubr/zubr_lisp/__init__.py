import os

LISP_LOCATION = os.path.abspath(os.path.dirname(__file__))
ZUBR_TOP = os.path.abspath(os.path.join(LISP_LOCATION,'../../'))
STDLIB = os.path.join(LISP_LOCATION,'stdlib')

def in_stdlib(path):
    expanded = os.path.join(STDLIB,path)
    if os.path.isfile(expanded):
        return expanded

def in_lispdir(path):
    expanded = os.path.join(LISP_LOCATION,path)
    if os.path.isfile(expanded):
        return expanded

def in_top(path):
    expanded = os.path.join(ZUBR_TOP,path)
    if os.path.isfile(expanded):
        return expanded 

def find_file(path):
    if os.path.isfile(path):
        return path

    stdlib_file = in_stdlib(path)
    lisp_file   = in_lispdir(path)
    on_top      = in_top(path)
    if stdlib_file:
        return stdlib_file
    elif lisp_file:
        return lisp_file
    elif on_top:
        return on_top

    raise IOError('unknown file: %s' % path)
