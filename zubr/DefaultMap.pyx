# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Custom cython implementation of a defaultdict, 
taken in large part from: http://code.activestate.com/recipes/523034-emulate-collectionsdefaultdict/ 

"""

cdef class DefaultIntegerMap(dict):
    """Emulates a defaultdict for integer and real values"""

    def __init__(self,*a,**kw):
        dict.__init__(self,*a,**kw)

    def __getitem__(self,key):
        try:
            return <double>dict.__getitem__(self,key)
        except KeyError:
            return <double>self.__missing__(key)

    def __missing__(self,key):
        cdef double value = 0.0
        self[key] = value
        return value

    def __repr__(self):
        return "DefaultIntegerMap(%s, %s)" % ("float",dict.__repr__(self))



# cdef class DefaultLongMap(dict):
#     """Emulates a defaultdict for integer and real values"""

#     def __init__(self,*a,**kw):
#         dict.__init__(self,*a,**kw)

#     def __getitem__(self,long key):
#         try:
#             return <double>dict.__getitem__(self,key)
#         except KeyError:
#             return <double>self.__missing__(key)

#     def __missing__(self,int key):
#         cdef double value = 0.0
#         self[key] = value
#         return value

#     def __repr__(self):
#         return "DefaultIntegerMap(%s, %s)" % ("float",dict.__repr__(self))
