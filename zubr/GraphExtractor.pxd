# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Feature extractor for graph decoders 

"""

import numpy as np
cimport numpy as np
from zubr.Extractor cimport Extractor
from zubr.Features cimport FeatureObj
from zubr.ZubrClass cimport ZubrSerializable
from zubr.FeatureComponents cimport WordPhraseComponents,KnowledgeComponents,RankComponents,StorageComponents
from zubr.Dataset cimport RankPair


cdef extern from "math.h":
    double log(double)
    double isfinite(double)
    double isinf(double)

cdef class GraphExtractorBase(Extractor):
    cdef public object _config
    cdef public object decoder
    cdef FeatureObj extract_from_scratch(self,RankPair instance,str etype)
    cdef FeatureObj extract_from_file(self,str directory, str etype,int identifier)

    ## extractor components 
    cdef WordPhraseComponents word_phrase
    cdef KnowledgeComponents knowledge
    cdef StorageComponents storage
    
cdef class GraphRankExtractor(GraphExtractorBase):
    cdef RankComponents ranks

