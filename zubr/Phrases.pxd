from zubr.Features cimport FeatureMap
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Datastructures cimport Sparse2dArray
import numpy as np
cimport numpy as np

cdef class TranslationSide:
    cdef public unicode string

cdef class HieroSide(TranslationSide):
    cdef public list nts
    cdef public unicode rule 
    cpdef int context_size(self)

cdef class PhraseSide(TranslationSide):
    pass 

## pairs

cdef class TranslationPair:
    cpdef bint sides_match(self)
    cpdef bint econtainsf(self)
    cpdef bint fcontainse(self)
    cpdef int word_overlap(self)
    cdef bint is_stop_word(self)
    
cdef class PhrasePair(TranslationPair):
    cdef public PhraseSide english
    cdef public PhraseSide foreign
    cdef public str lang
    cdef int num
    cdef int eid,fid

cdef class HieroRule(TranslationPair):
    cdef public unicode lhs
    cdef public HieroSide erhs,frhs
    cdef public int freq
    cdef public int rule_number
    cdef public str lang
    cdef int eid,fid

    ## hiero comparison functions 
    cpdef bint has_reordering(self)
    cpdef bint left_contexts_match(self)
    cpdef bint right_contexts_match(self)
    cpdef bint only_terminals(self)
    cdef  bint is_stop_word(self)
    cpdef bint left_terminal_only(self)
    cpdef bint right_terminal_only(self)

## c methods

cdef PhrasePair ConstructPair(unicode s1, unicode s2,str lang=?)    
cdef HieroRule  ConstructHPair(unicode lhs,unicode s1,unicode s2,str lang=?)

## phrase tables

cdef class PhraseTableBase(ZubrSerializable):
    cdef public np.ndarray slist
    cdef public np.ndarray phrases
    cdef public int elen,flen
    cdef np.ndarray ids
    cpdef int query(self,input1,input2,lhs=?)
    
cdef class SimplePhraseTable(PhraseTableBase):
    cdef PhrasePair create_pair(self,input1,input2,str lang=?)

cdef class HieroPhraseTable(PhraseTableBase):
    cdef public dict glue
    cdef public dict lhs_lookup
    cdef HieroRule create_rule(self,input1,input2,lhs,str lang=?)

cdef class ParaphraseTable(PhraseTableBase):
    pass

cdef class Sparse2dWordPairs(Sparse2dArray):
    pass

cdef class SparseDictWordPairs(ZubrSerializable):
    cdef int find_identifier(cls,f,e)
    cdef public dict word_dict

cdef class DescriptionWordPairs(SparseDictWordPairs):
    pass
