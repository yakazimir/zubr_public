
from zubr.Extractor cimport Extractor
#from zubr.SymmetricAligner cimport M1SymmetricAligner
from zubr.Dataset cimport RankDataset,RankStorage,RankPair
from zubr.Features cimport FeatureObj

cdef extern from "math.h":
    double log(double)
    double isfinite(double)
    double isinf(double)

cdef class AlignerExtractor(Extractor):
    cdef public object _config
    cdef public object base_model
    cdef RankStorage trainranks,testranks,validranks,queryranks
    cdef FeatureObj extract_from_scratch(self,RankPair instance,str etype)
    cdef FeatureObj extract_from_file(self,str directory,str etype, int identifier)
