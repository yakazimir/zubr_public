from zubr.Extractor cimport Extractor
from zubr.Features cimport FeatureObj
from zubr.Dataset cimport RankPair,RankStorage

cdef class BowExtractor(Extractor):
    cdef public object _config
    cdef RankStorage trainranks,testranks,validranks
    cdef FeatureObj extract_from_scratch(self,RankPair instance,str etype)
    cdef FeatureObj extract_from_file(self,str directory,str etype, int identifier)
