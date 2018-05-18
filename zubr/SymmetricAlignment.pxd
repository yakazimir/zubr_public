#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementation of a symmetric aligner using aligners from Aligner.pyx 
"""

from zubr.Aligner cimport Model1,Model2,Alignment
from zubr.Alignment cimport WordModel
from zubr.ZubrClass cimport ZubrSerializable
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    bint isinf(double)
    bint isfinite(double)
    double log(double)

## create alignment from giza string
cdef np.ndarray from_giza(unicode giza_string,int elen,int flen,int estart,int eend,bint reverse=?)

cdef class SymmetricAlignerBase(ZubrSerializable):
    cpdef int train(self,object config) except -1
    cpdef SymAlign align(self,f,e,str heuristic=?)
    cpdef int align_dataset(self,object config) except -1
    cdef SymAlign _align(self,int[:] e,int[:] f,int[:] e2,str heuristic=?)
    cpdef int phrases_from_data(self,object config) except -1
    cpdef int extract_hiero(self, object config) except -1
    cdef public dict phrase_table
    #cpdef tuple rank_dataset(self,object config)
    cdef inline int pair_identifier(self,int eid,int fid, int flen)


cdef class SymmetricWordModel(SymmetricAlignerBase):
    cdef public WordModel etof
    cdef public WordModel ftoe
    #cpdef tuple _return_viterbi(self,foreign,english,heuristic=?)
    
cdef class SymAlign:
    cdef public np.ndarray alignment
    cdef public np.ndarray union_alignment
    cdef int elen
    cdef int flen
    cdef int heuristic
    cdef double prob
    cdef void make_symmetric(self,int[:] etof,int[:] ftoe)
    cpdef unicode giza(self)
    cdef Phrases extract_phrases(self,int max_len)
    cpdef Phrases phrases(self,int max_len)

cdef class Phrases:
    cdef public np.ndarray phrases
    cdef void reduce_phrase(self,int size)
    cpdef list phrase_list(self,e,f)
    cpdef list lexical_positions(self,int[:] f,int[:] e)

