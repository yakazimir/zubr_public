import numpy as np
cimport numpy as np
from zubr.DefaultMap cimport DefaultIntegerMap
from zubr.ZubrClass cimport ZubrSerializable

cdef extern from "math.h":
    bint isinf(double)
    bint isnan(double)
    bint isfinite(double)
    double log(double)
    double exp(double)

cdef class FeatureCounter(dict):
    pass

cdef class TranslationCounter(FeatureCounter):
    pass

cdef class Vectorizer:
    cdef public np.ndarray feat_counts
    cdef public np.ndarray features

cdef class TemplateManager(dict):
    cdef public dict description
    cdef public dict starts

cdef class FeatureObj(ZubrSerializable):
    cdef list _features
    cdef public int beam
    cdef public int baseline
    cdef FeatureMap _gold_features 
    cdef Vectorizer vectorize_item(self,int i)
    cdef Vectorizer vectorize_gold(self)
    cpdef void print_features(self,wdir,ftype,identifier,rvals,gold)
    cpdef void create_binary(self,wdir,ftype,identifier)

cdef class BinnedFeatures(DefaultIntegerMap):
    pass

cdef class FeatureMap(DefaultIntegerMap):
    cdef dict _templates
    cdef long maximum
    cdef BinnedFeatures _binned
    cdef int add_binary(self,int vindx,unsigned long increm, double value=?) except -1
    cdef void add_incr(self,int vindx,unsigned long increm, double value)
    cdef void add_binned(self,int vindx)
    cdef void compute_neg_bins(self,int[:] negative,double threshold)
    #cdef void load_from_string(self,str feature_input)
    cdef int load_from_string(self,str feature_input) except -1
    cdef int add_internal(self,int vindx, unsigned long offset,unsigned long increm,double value=?) except -1
    cdef int add_increm_binary(self,int vindx,unsigned long increm, double value=?) except -1

cdef class FeatureAnalyzer(ZubrSerializable):
    cdef int size
    cdef double first_rank
    cdef public double likelihood
    cdef public np.ndarray probs
    cdef public dict feature_scores
    cdef double averaged_nonzeroed
    cdef double gold_prob(self)
    cdef void add_gold_feature(self,long identifier,double value,double score)
    cdef void add_feature(self,int num, long identifier,double value,double score)
    cdef int normalize(self,int start=?,int end=?) except -1

## empty

cdef BinnedFeatures EMPTY_BIN
cdef TemplateManager EMPTY_TEMPLATE
