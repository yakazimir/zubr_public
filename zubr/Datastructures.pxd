#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementations of common datastructures

"""

import numpy as np
cimport numpy as np
from zubr.ZubrClass cimport ZubrLoggable,ZubrSerializable

ENCODING = 'utf-8'

cdef inline unicode to_unicode(input_str):
    if isinstance(input_str,bytes):
        return (<bytes>input_str).decode(ENCODING)
    return input_str

## SPARSE ARRAY

cdef class Sparse2dArray(ZubrSerializable):
    cdef np.ndarray dim1,dim2,spans
    cdef int find_id(self,int index1,int index2)

cdef class SuffixArrayBase(ZubrLoggable):
    pass

cdef class StringSuffixArray(SuffixArrayBase):
    cdef unicode orig_input
    cdef int size
    cdef np.ndarray sorted_array,prefix_array

cdef class SentenceSuffixArray(SuffixArrayBase):
    pass

cdef class ListLookupArray(SuffixArrayBase):
    pass 

cdef class ParallelCorpusSuffixArray(SuffixArrayBase):
    cdef np.ndarray ecorpus,fcorpus
    cdef np.ndarray esorted,fsorted
    cdef np.ndarray esen,fsen,starts
    cdef public list alignment
    cdef public int size
    cdef public str name
    #cpdef unicode return_alignment(self,int sentence_num)

cdef class Indexer:
    cdef public int start,end,size
