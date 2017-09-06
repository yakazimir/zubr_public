# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""

from zubr.ZubrClass cimport ZubrSerializable
from zubr.Alignment cimport WordModel
#from zubr.NeuralModels cimport FeedForwardTranslationLM

cdef class RankerBase(ZubrSerializable):
    cpdef int rank(self,object config) except -1

cdef class RankDecoder(RankerBase):
    cdef WordModel aligner

cdef class MonoRankDecoder(RankDecoder):
    pass

cdef class PolyglotRankDecoder(RankDecoder):
    pass

## neural rankers

# cdef class NeuralRankDecoder(RankerBase):
#     cdef public FeedForwardTranslationLM model

# cdef class NeuralSingleDecoder(NeuralRankDecoder):
#     pass

