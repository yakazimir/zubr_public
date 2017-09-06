#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

cython re-implementation of Norvig's lispy interpreter 
"""

import re
import os
from zubr.zubr_lisp.Reader import LispSymbol,LispReader,make_atomic
from zubr.zubr_lisp.Reader import _quote,_quasiquote,_unquote,_unquotesplicing,quotes
from zubr.zubr_lisp.Core import _global_vars as lglobals
from zubr.zubr_lisp import find_file
from types import FunctionType
from zubr.util.loader import load_module,load_script
import inspect

ENCODING = 'utf-8'

## unicode helper function

cdef unicode to_unicode(input_str):
    if isinstance(input_str,bytes):
        return (<bytes>input_str).decode(ENCODING)
    return input_str

## primitive expression symbols

_define       = LispSymbol(u'define')
_begin        = LispSymbol(u'begin')

## quotes (defined in reader) 

_if           = LispSymbol(u'if')
_set          = LispSymbol(u'set!')
_lambda       = LispSymbol(u'lambda')
_definemacro  = LispSymbol(u'define-macro')

## class special forms 
_defineclass  = LispSymbol(u'define-class')
_definerecord = LispSymbol(u'define-record')
_attr         = LispSymbol(u'attr')
_functions    = LispSymbol(u'functions')

## control structures
_let          = LispSymbol(u'let')
_cons         = LispSymbol(u'cons')
_append       = LispSymbol(u'append')
_cond         = LispSymbol(u'cond')
_try          = LispSymbol(u'try')
_catch        = LispSymbol(u'catch')
_globalrecord = LispSymbol(u'global-record')

## procedure class

class Procedure:

    """main class for defining lambda functions"""
    
    def __init__(self,params,exp,env):
        self.params = params
        self.exp    = exp
        self.env    = env

    def __call__(self,*args):
        return evaluate(self.exp,Environment(self.params,args,self.env))

## environment class

class Environment(dict):

    """Environment and namespace object"""
    
    def __init__(self,params=(),args=(),outer=None):
        self.outer = outer
        if isinstance(params,LispSymbol):
            self.update({params:list(args)})
        else:
            self.update(zip(params,args))

    def find(self,variable):
        """find items in the enviornment/namespace

        :param variable: the variable to find
        :type variable: str
        :returns: self
        """
        if variable in self:
            return self
        elif self.outer is None:
            raise LookupError('unknown symbol: %s' % variable)
        else:
            return self.outer.find(variable)

def create_record(name,subclass,attrs):
    """dynamically creates a record (method-less class) using ``type`` 

    :param name: name of the class
    :param subclass: class subtype
    :param attrs: instance attributes
    """
    record_attrs = {'_cname':name}
    if subclass != u'object':
        record_attrs.update({i for i in subclass.__dict__.iteritems() if i[0][:1] != '_'})
    else:
        subclass = object 
    
    attrs = dict(attrs) if attrs else {}
    record_attrs.update(attrs)
    cl = type(str(name),(subclass,),record_attrs)
    return cl

global_env = Environment()
global_env.update(lglobals)

### evaluate function

def evaluate(minput,environment=global_env,make_global=True):
    """The main lisp evaluation function

    :param minput: the input to evaluate in python lisp form
    """

    while True:

        ## zubr symbol
        if isinstance(minput,LispSymbol):
            return environment.find(minput)[minput]
        
        ## anything besides a list
        elif not isinstance(minput,list):
            return minput

        elif minput[0] == _quote:
            (_,exp) = minput
            return exp

        ## conditional control structure 
        elif minput[0] == _if:
            (_,test,conseq,alt) = minput
            minput = (conseq if evaluate(test,environment) else alt)

        ## arbitrary conditional
        elif minput[0] == _cond:
            conditions = minput[1:]
            for (condition,result) in conditions[:-1]:
                if evaluate(condition,environment):
                    return evaluate(result,environment)
            return evaluate(conditions[-1],environment)

        ## setting a variable 
        elif minput[0] == _set:
            (_,var,exp) = minput
            environment.find(var)[var] = evaluate(exp,environment)
            return None

        # ## attr item witn :name
        elif isinstance(minput[0],basestring) and re.search(r'^\:',minput[0]):
            if len(minput) > 2:
                raise SyntaxError('attribute syntax not correct')

            attr = minput[0].split(':')[-1]
            obj = evaluate(minput[1],environment)

            ## is a dictionary
            if isinstance(obj,dict):
                return obj[attr]

            # class 
            elif isinstance(obj,type):
                return obj.__dict__[attr]
            else:
                raise ValueError('Wrong input type for Attribute (:) function: %s' % type(obj))
            
        ## creating a definition
        elif minput[0] == _define:
            (_,var,exp) = minput
            if not make_global:
                return (str(var),evaluate(exp,environment)) 
            environment[var] = evaluate(exp,environment)
            return None

        ## global class
        elif minput[0] == _globalrecord:
            (_,record) = minput
            rclass = evaluate(record,environment)
            environment[rclass._cname] = rclass
            return
        
        ## create a ``record`` (i.e., method-less class)
        elif minput[0] == _definerecord:
            if len(minput[1:]) < 3:
                raise SyntaxError('malformed record definition')

            cname,subclass,attrs = minput[1:]
            
            if isinstance(cname,list):
                cname = evaluate(cname,environment)

            # if subclass != u'object' and not environment.get(subclass):
            if subclass != u'object' and not global_env.get(subclass):
                raise ValueError('unknown subclass: %s' % subclass)

            #subclass = subclass if not environment.get(subclass) else environment.get(subclass)
            subclass = subclass if not global_env.get(subclass) else global_env.get(subclass)
            attrs = [(str(n[0]),evaluate(n[1],environment)) for n in attrs[1:] if n]
            new_class = create_record(cname,environment.get(subclass,subclass),attrs)
            #environment[cname] = new_class
            
            return new_class

        ## creating a lambda/anonymous function
        elif minput[0] == _lambda:
            (_,vars,exp) = minput
            return Procedure(vars,exp,environment)

        ## begin statement
        elif minput[0] == _begin:
            for exp in minput[1:-1]:
                evaluate(exp, environment)
            minput = minput[-1]


        ## causes many issues
        
        ## error exception
        # if minput[0] == _try:
        #     to_try = minput[1]
        #     catches = minput[2]
        #     try:
        #         return evaluate(to_try,environment)
            
        #     except Exception,e:

        #         ## check through provided exceptions
        #         for catch in catches[1:]:
        #             exception_type,command = catch
        #             exception_type = evaluate(exception_type,environment)
        #             if isinstance(e,exception_type):
        #                 return evaluate(command,environment)
        #     finally: 
        #         raise(e)

        ## execute function 
        else:
            exps = [evaluate(exp,environment) for exp in minput]
            proc = exps.pop(0)
            if isinstance(proc,Procedure):
                defines_class = False

                ## if this defines a class, stores subclass in environment
                if proc.exp and _definerecord == proc.exp[0]:
                    subclass_name = proc.exp[2]
                    #subclass_value = environment[subclass_name]
                    subclass_value = global_env[subclass_name]
                    defines_class = True
                                                                                          
                minput = proc.exp
                environment = Environment(proc.params,exps,proc.env)
                if defines_class:
                    environment[subclass_name] = subclass_value                                 

            else:
                return proc(*exps)

## taken directly from norvig

def is_pair(x):
    return x != [] and isinstance(x, list)

def cons(x, y):
    return [x]+y

def require(x, predicate, msg="wrong length"):
    "Signal a syntax error if predicate is false."
    if not predicate:
        raise SyntaxError(x+': error'+msg)

def expand_quasiquote(x):
    """Expand `x => 'x; `,x => x; `(,@x y) => (append x y) """
    if not is_pair(x):
        return [_quote, x]
    require(x, x[0] != _unquotesplicing, "can't splice here")
    if x[0] == _unquote:
        require(x, len(x)==2)
        return x[1]
    elif is_pair(x[0]) and x[0][0] == _unquotesplicing:
        require(x[0], len(x[0])==2)
        return [_append, x[0][1], expand_quasiquote(x[1:])]
    else:
        return [_cons, expand_quasiquote(x[0]), expand_quasiquote(x[1:])]

def let(*args):
    args = list(args)
    x = cons(_let, args)
    require(x, len(args) > 1)
    bindings, body = args[0], args[1:]
    require(x, all(isinstance(b, list) and len(b)==2 and isinstance(b[0], LispSymbol)
                   for b in bindings), "illegal binding list")
    vars, vals = zip(*bindings)
    return [[_lambda, list(vars)]+map(expand, body)] + map(expand, vals)


macro_table = {_let:let} ## More macros can go here

def expand(x, toplevel=False):
    "Walk tree of x, making optimizations/fixes, and signaling SyntaxError."

    require(x, x!=[])                    # () => Error

    # constant => unchances
    if not isinstance(x, list):
        return x

    # (quote exp)
    elif x[0] == _quote:
        require(x, len(x)==2)
        return x

    # (if t c) => (if t c None) 
    elif x[0] == _if:                    
        if len(x)==3: x = x + [None]
        require(x, len(x)==4)
        return map(expand, x)
    
    elif x[0] == _set:                   
        require(x, len(x)==3); 
        var = x[1]                       # (set! non-var exp) => Error
        require(x, isinstance(var, LispSymbol), "can set! only a symbol")
        return [_set, var, expand(x[2])]

    elif x[0] == _define or x[0] == _definemacro: 
        require(x, len(x)>=3)            
        _def, v, body = x[0], x[1], x[2:]
        if isinstance(v, list) and v:           # (define (f args) body)
            f, args = v[0], v[1:]        #  => (define f (lambda (args) body))
            return expand([_def, f, [_lambda, args]+body])
        else:
            require(x, len(x)==3)        # (define non-var/list exp) => Error
            require(x, isinstance(v, LispSymbol), "can define only a symbol")
            exp = expand(x[2])
            if _def == _definemacro:     
                require(x, toplevel, "define-macro only allowed at top level")
                proc = evaluate(exp)       
                require(x, callable(proc), "macro must be a procedure")
                macro_table[v] = proc    # (define-macro v proc)
                return None              #  => None; add v:proc to macro_table
            return [_define, v, exp]

    ## progn or begin statement
    elif x[0] == _begin:
        if len(x)==1: return None        # (begin) => None
        else: return [expand(xi, toplevel) for xi in x]

    ## lambda definition 
    elif x[0] == _lambda:
        require(x, len(x)>=3)
        vars, body = x[1], x[2:]
        require(x, (isinstance(vars, list) and all(isinstance(v, LispSymbol) for v in vars))
                or isinstance(vars, LispSymbol), "illegal lambda argument list")
        exp = body[0] if len(body) == 1 else [_begin] + body
        return [_lambda, vars, expand(exp)]   

    # `x => expand_quasiquote(x) 
    elif x[0] == _quasiquote:
        require(x, len(x)==2)
        return expand_quasiquote(x[1])

    # lisp symbol 
    elif isinstance(x[0], LispSymbol) and x[0] in macro_table:
        return expand(macro_table[x[0]](*x[1:]), toplevel) # (m arg...)
  
    #  => macroexpand if m isa macro
    else:
        return map(expand, x)            # (f arg...) => expand each


## query the evaluator

def query(input):
    minput =  to_unicode(input)
    parsed = expand(LispReader.parse(minput)[0],toplevel=True)
    out = evaluate(parsed)
    return out


def query_from_list(input):
    parsed = expand(input,toplevel=True)
    return evaluate(parsed)

def parse(input):
    minput = to_unicode(input)
    return expand(LispReader.parse(minput)[0],toplevel=True)

## create standard global environment

## load functions

def load_file(file_path):
    """load a zubr lisp source file

    :param file_path: the path of the lisp file
    :type file_path: str
    """
 
    file_location = find_file(file_path)
    contents = LispReader.parse(file_location)
    for item in contents:
        m = expand(item,toplevel=True)
        evaluate(m,environment=global_env)


def use_package(package_name):
    pass

def load_python_script(path):
    script = load_script(path)
    functions = inspect.getmembers(script,inspect.isfunction)
    for function_name,impl in functions:
        global_env[function_name] = impl


global_env['load'] = load_file
global_env['use-package'] = use_package
global_env['py-script']   = load_python_script


STDLIB = [
    "core.lisp",
    "macros.lisp",
    "list.lisp",
    "dict.lisp",
    "map.lisp",
    "records.lisp",
    "bool.lisp",
    "assert.lisp",
    "curry.lisp",
]

## load the standard macros

    
for stdlib_item in STDLIB:
    query("""(load "%s")""" % stdlib_item)
    
#query('(load "macros.lisp")')
#query('(load "records.lisp")')

## core stdlib files 
## query('(load "core.lisp")')
