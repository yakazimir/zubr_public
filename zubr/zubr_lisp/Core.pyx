from cython cimport cdivision
import numpy as np
cimport numpy as np
import operator
import codecs
import sys
import re
from zubr.zubr_lisp.Reader import LispSymbol

ENCODING = 'utf-8'

cpdef unicode to_unicode(input_str):
    if isinstance(input_str,bytes):
        return (<bytes>input_str).decode(ENCODING)
    return input_str

cdef extern from "math.h":
    double log(double)
    bint isfinite(double)
    bint isinf(double)
    bint isnan(double)
    double sin(double x)
    double cos(double x)

def display(to_display):
    port = sys.stdout
    port.write(str(to_display)+"\n")
    #sys.stdout.flush()

cdef bint is_boolean(object value):
    return isinstance(value,'bool')

cdef bint is_list(object value):
    return isinstance(value,list)

cdef bint is_set(object value):
    return isinstance(value,set)

cdef object car(list input_list):
    return input_list[0]

cdef list cdr(list input_list):
    return input_list[1:]

def chain_list(list_of_lists):
    return sum(list_of_lists,[])

def list_overlap(x,y):
    return [i for i in x if i in y]

# cdef unicode concat(object str1, object str2):
#     return <unicode>to_unicode(str1)+to_unicode(str2)

def concat(*args):
    return ''.join([str(a) for a in args])

cdef double _log(double x) except -1:
    return log(x)

cdef bint _isfinite(double x) except -1:
    return isfinite(x)

cdef bint _isinfinite(double x) except -1:
    return isinf(x)

cdef bint _isnan(double x) except -1:
    return isnan(x)

cdef double _sin(double x) except -1:
    return sin(x)

cdef double _cos(double x) except -1:
    return cos(x)

cdef object _get_item(list x,int index):
    return x[index]

cdef bint equality(object x,object y) except -1:
    return x == y

cdef bint contains(object x,list y) except -1:
    return <bint>(x in y)

## dictionary methods

cdef list key_list(dict x):
    return <list>x.keys()

cdef list val_list(dict x):
    return <list>x.values()

def get_val(x,y,default=True):
    try:
        return x[y]
    except KeyError,e:
        if default:
            raise e
        return None

def cons(x,y):
    return [x]+y

def null(x):
    return x == []

def make_list(*x):
    return list(x)

def make_set(*x):
    return set(x)

def is_symbol(x):
    return isinstance(x,LispSymbol)

def read_file(path,encoding=ENCODING):
    return codecs.open(path,encoding=ENCODING)

def read_lines(open_file):
    return open_file.readlines()

## record methods

def get_class_attr(attr,rtype):
    return rtype.__dict__[attr]

def set_class_attr(attr,rtype,val):
    setattr(rtype, attr, val)

def sub_classes(zclass):
    return [c.__name__ for c in getattr(zclass,'__subclasses__')()]

def make_string(value):
    return to_unicode(str(value))

def make_split(string,delim=u' '):
    return string.split(delim)

def update_dict(x,y):
    z = x.copy()
    z.update(y)
    return z

def join(mlist,delim=' '):
    return delim.join(mlist)

def format_string(pattern,*string):
    return pattern.format(*string)

def add_newline(value):
    return "%s\n" % value

## raising exceptions 

def raise_exception(exception=Exception,msg=''):
    raise exception(msg)

## callable 
def callable_item(item):
    return hasattr(item,"__call__")

def is_object(item):
    return isinstance(item,type)

## global functions

_global_vars = {
    'type'        : type,
    'isinstance'  : isinstance,
    ## comparison
    '+'           : operator.add,
    '-'           : operator.sub,
    'not'         : operator.not_,
    '>'           : operator.gt,
    '<'           : operator.lt,
    '>='          : operator.ge,
    '<='          : operator.le,
    '='           : operator.eq,
    'null?'       : null,
    '*'           : operator.mul,
    'add'         : operator.add,
    'subtract'    : operator.sub,
    'divide'      : operator.div,
    '/'           : operator.div,
    'multiply'    : operator.mul,
    #'sqrt'        : '',
    ## list methods
    'list'       : make_list,
    'car'        : car,
    'cdr'        : cdr,
    'len'        : len,
    'length'     : len,
    'map'        : map,
    'filter'     : filter,
    'reduce'     : reduce,
    'nth'        : _get_item,
    'sum'        : sum,
    'range'      : range,
    'append'     : operator.add,
    'zip'        : zip,
    'contains'   : contains,
    #'chain'      : chain_lists,
    'list-overlap':list_overlap,
    ### predicates    
    'set?'       : is_set,
    'boolean?'   : is_boolean,
    'list?'      : is_list,
    'atom?'      : '',
    'eq?'        : operator.is_,
    'isfinite?'  : _isfinite,
    'isinf?'     : _isinfinite,
    'isnan?'     : _isnan,
    'symbol?'    : is_symbol,
    ## string
    'concat'     : concat,
    'cons'       : cons,
    'string'     : make_string,
    'split'      : make_split,
    'join'       : join,
    'format-string': format_string,
    ## general 
    'set'        : make_set,
    'set-obj'    : set,
    'len'        : len,
    'map'        : map,
    'reduce'     : reduce,
    ## python dictionary/map implementation
    'dictionary' : dict,
    'dict'       : dict,
    'keys'       : key_list,
    'vals'       : val_list,
    'get-val'    : get_val,
    'update'     : update_dict,
    ### python class/record implementation
    'r-attr'    : get_class_attr,
    's-attr'    : set_class_attr,
    'rdir'      : dir,
    'subclasses': sub_classes,
    ## ndarray from numpy 
    #'ndarray'    :
    ## c level numeric stuff
    'log'        : _log,
    'sin'        : _sin,
    'cos'        : _cos,
    ## file, i/o
    'open-file'  : read_file,
    'readlines'  : read_lines,
    ## make numbers
    'float'      : float,
    'int'        : int,
    'unicode'    : unicode,
    'basestring' : basestring,
    'complex'    : complex,
    ## conjunction
    #'and'        : and,
    'print'      : display,
    'display'    : display,
    'newline'    : add_newline,
    ## exceptions
    'Exception'  : Exception,
    'ValueError' : ValueError,
    'IndexError' : IndexError,
    'AttributeError':AttributeError,
    'TypeError'  : TypeError,
    'KeyError'   : KeyError,
    'raise'      : raise_exception,
    ## max,min
    'max' : max,
    'min' : min,
    ## assertion
    #'assert' : assert,
    'callable?' : callable_item,
    'is-record?'   : is_object,
    'isinstance': isinstance,
    }

