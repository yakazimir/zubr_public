# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package. 

author : Kyle Richardson
"""

import numpy as np
cimport numpy as np
from zubr.ZubrClass cimport ZubrSerializable

cdef class Pair:
    cdef np.ndarray en,rep

cdef class RankPair:
    cdef public np.ndarray en
    cdef public int rep_id
    cdef public int global_id
    cdef public unicode surface
    cdef public object lang

cdef class Data(ZubrSerializable):
    pass 

cdef class EmptyDataset(Data):
    pass

cdef class Dataset(Data):
    cdef void shuffle(self)
    cdef RankPair next_ex(self)
    cdef unsigned int _size,_index
    cdef public np.ndarray _dataset_order
    cpdef RankPair get_item(self,int index)
    
cdef class RankDataset(Dataset):
    cdef public np.ndarray en
    cdef public np.ndarray rank_index
    cdef public np.ndarray en_original
    cdef public np.ndarray langs

## rank evaluator

cdef class RankEvaluator:
    cdef int size,rank_size
    cdef str rtype

## rank storage

cdef class RankStorage(ZubrSerializable):
    cdef np.ndarray ranks
    cdef np.ndarray gold_pos
    cdef dict other_gold
    cdef np.ndarray find_ranks(self,int indx,str ttype,int beam)
    cdef int gold_position(self,int indx)
    cdef int gold_value(self,int indx)
    cdef np.ndarray empty_rank_list(self)

cdef RankStorage EMPTY_STORAGE

## rank comparison

cdef class RankComparison(ZubrSerializable):
    cdef RankStorage storage
    cdef np.ndarray new_ranks
    cdef int beam
    cdef np.ndarray rank_example(self,int i)
    cdef np.ndarray old_ranks(self,int i)
    
    ## functions for rerrank scoring 
    cdef RankScorer evaluate(self,str ttype,str wdir,int it=?,double ll=?,bint debug=?)
    cdef RankScorer multi_evaluate(self,RankDataset dataset,str ttype,str wdir,int it=?,double ll=?,bint debug=?)

## scoring functions

cdef class Scorer:
    cpdef bint less_than(self,other)
    cpdef bint greater_than(self,other)
    cpdef double score(self)

cdef class RankScorer(Scorer):
    cdef public double at1
    cdef public double at10
    cdef public double mrr
