from zubr.Phrases cimport HieroPhraseTable,SimplePhraseTable,SparseDictWordPairs,DescriptionWordPairs
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Dataset cimport RankDataset,RankStorage,RankPair
import numpy as np
cimport numpy as np

## WORD PHRASE COMPONENTS 

cdef class WordPhraseComponents(ZubrSerializable):
    cdef HieroPhraseTable hiero
    cdef SimplePhraseTable phrases
    cdef SparseDictWordPairs pairs

## RANK COMPONENTS

cdef class RankComponents(ZubrSerializable):
    cdef np.ndarray rank_list,trees,rank_vals
    cdef dict classes,langs
    cdef int[:] rank_item(self,int index)
    cdef unicode surface_item(self,int index)
    cdef int language_id(self,int index)
    
cdef class PolyComponents(RankComponents):
    pass

cdef class MonoComponents(RankComponents):
    pass

## KNOWLEDGE COMPONENTS

cdef class KnowledgeComponents(ZubrSerializable):
    cdef DescriptionWordPairs descriptions

## STORAGE COMPONENTS

cdef class StorageComponents(ZubrSerializable):
    cdef RankStorage trainranks,testranks,validranks,queryranks
