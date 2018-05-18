# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""

from zubr.Aligner cimport Model1
import numpy as np
cimport numpy as np

cdef class BaseQueryObj:
    cpdef list query(self,qinput,int size=?)
    cdef inline unicode to_unicode(self,s)
    cdef str _name,encoding
    cdef bint lower

cdef class M1Query(BaseQueryObj):
    cdef Model1 aligner
    cdef np.ndarray ranks
    cdef np.ndarray _query(self,int[:] en,int size)
    cdef list rank_html #rank_str
