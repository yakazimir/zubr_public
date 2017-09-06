# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""

from zubr.Optimizer cimport RankOptimizer
from zubr.Dataset cimport RankScorer,RankComparison
from zubr.ZubrClass cimport ZubrLoggable
from zubr.Optimizer cimport RankOptimizer
from zubr.Extractor cimport Extractor

cdef class FeatureSelector(ZubrLoggable):
    cdef RankScorer current_score

cdef class GreedyWrapperSelector(FeatureSelector):
    cdef RankOptimizer model
    cpdef int select(self) except -1
    cdef int selection_loop(self) except -1

cdef class ForwardSearch(GreedyWrapperSelector):
    pass 

cdef class BackwardSearch(GreedyWrapperSelector):
    cdef object config
