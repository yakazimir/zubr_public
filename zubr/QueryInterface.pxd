# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Classes for querying rank models 

"""

from zubr.ZubrClass cimport ZubrSerializable
import numpy as np
cimport numpy as np

cdef class BaseQueryInterface(ZubrSerializable):
    cpdef QueryOutput query(self,query_input,int size=?)
    cpdef np.ndarray encode_input(self,rinput)

cdef class RankDecoderInterface(BaseQueryInterface):
    cdef np.ndarray _rank_reps
    cdef int rank_size

cdef class TranslationModelInterface(RankDecoderInterface):
    cdef object rank_decoder

cdef class RerankerDecoderInterface(RankDecoderInterface):
    cdef object extractor, model

cdef class HTMLRerankerInterface(RerankerDecoderInterface):
    pass

cdef class TextRerankerInterface(RerankerDecoderInterface):
    pass 

## helper class for passing query interface output

cdef class QueryOutput:
    cdef public object rep
    cdef public double time
    cdef public list ids
