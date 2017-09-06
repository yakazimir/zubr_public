# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Base classe for all feature extractors 

To implement a new feature extractor, inherit from ExtractBase and 
add class to EXTRACTORS map in  FeatureExtractor.pyx 

"""

from zubr.ZubrClass cimport ZubrSerializable
from zubr.Features cimport FeatureObj
from zubr.Dataset cimport RankPair,RankComparison,RankDataset
import numpy as np
cimport numpy as np


cdef class ExtractorBase(ZubrSerializable):
    cpdef void offline_init(self,object dataset,str rtype)
    cdef FeatureObj extract(self,RankPair instance,str etype)
    cdef RankComparison rank_init(self,str etype)
    cdef FeatureObj extract_query(self,RankPair instance,str etype)
    cdef void after_eval(self,RankDataset dataset,RankComparison ranks)

cdef class Extractor(ExtractorBase):
    ## should have a config object 
    pass
