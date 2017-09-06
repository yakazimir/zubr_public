#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementations of the ``pivot-based`` bi-lingual paraphrase model using
Bannard and Callison-burch method

The method uses the suffix-array datastructure implemented in Zubr.Datastructures

"""
import numpy as np
cimport numpy as np
from zubr.ZubrClass cimport ZubrSerializable
from zubr.SymmetricAlignment cimport SymmetricWordModel,SymAlign,Phrases


cdef class ParaphraseBase(ZubrSerializable):
    cpdef list find_paraphrases(self,input_query,top_k=?)

cdef class MultilingualParaphraser(ParaphraseBase):
    pass

cdef class PivotParaphraser(MultilingualParaphraser):
    cdef public list corpora
    cdef bint lowercase
    cdef int sample_size

cdef class CounterBase:
    cdef set agenda
    cdef dict corpus_occ,phrase_occ
    cdef unicode pop(self)
    
cdef class PhraseCounter(CounterBase):
    cdef double occ 
    cdef void add_phrase(self,int cnum,unicode phrase)
    cdef void add_agenda(self,unicode phrase)

cdef class ForeignPhraseCounter(CounterBase):
    cdef void add_foreign(self,int cnum, unicode candidate, unicode trigger)
    
cdef class ParaphraseManager:
    cdef PhraseCounter ecounter
    cdef ForeignPhraseCounter fcounter
    cdef int num_corpora
    cdef list compute_probabilities(self,int maxsize)
    
cdef class CorpusParaphraser(ZubrSerializable):
    cdef SymmetricWordModel model
    cdef MultilingualParaphraser paraphraser
    cdef object lm

cdef class PhraseParaphraser(CorpusParaphraser):
    pass
    
cdef class GrammarParaphraser(CorpusParaphraser):
    pass


