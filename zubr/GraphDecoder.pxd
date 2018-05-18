# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson


A simple graph decoder for finite languages, using networkx 

"""
from zubr.ZubrClass cimport ZubrSerializable
from zubr.SymmetricAlignment cimport SymmetricWordModel
from zubr.Graph cimport WordGraph,DirectedAdj,KShortestPaths,Path
from zubr.ExecutableModel cimport ExecutableModel
from zubr.SymmetricAlignment cimport SymmetricWordModel,SymAlign,Phrases
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    bint isnan(double)
    bint isinf(double)
    
cdef class GraphDecoderBase(ZubrSerializable):
    cpdef int decode_data(self,object config) except -1

cdef class WordModelDecoder(GraphDecoderBase):
    cdef SymmetricWordModel model
    cdef KShortestPaths decode_input(self,int[:] decode_input,int k)
    #cpdef int decode_data(self,object config) except -1
    cdef WordGraph graph
    cdef np.ndarray edge_labels
    cdef dict edge_map
    cdef np.ndarray _encode_input(self,text_input)
    cpdef KBestTranslations translate(self,dinput,int k)
    cdef SymAlign align(self,int[:] output_v,int[:] input_v,str heuristic)

## rank decoders 
    
cdef class WordGraphDecoder(WordModelDecoder):
    pass #cdef np.ndarray _weight

cdef class PolyglotWordDecoder(WordModelDecoder):
    cdef dict langs
    
## executable decoder

cdef class ExecutableGraphDecoder(WordModelDecoder):
    cdef ExecutableModel executor

cdef class ExecutablePolyGraphDecoder(ExecutableGraphDecoder):
    pass

## neural decoders

cdef class NeuralShortestPath(WordModelDecoder):
    pass

cdef class PolyglotNeuralShortestPath(WordModelDecoder):
    pass

## concurrent decoders

cdef class ConcurrentWordModelDecoder(WordModelDecoder):
   cpdef _setup_jobs(self,config)

cdef class PolyglotConcurrentDecoder(ConcurrentWordModelDecoder):
    cdef dict langs

cdef class WordGraphConcurrentDecoder(ConcurrentWordModelDecoder):
    pass #cdef np.ndarray _weight ## is this needed?

## utility classes 

cdef class SequencePath(Path):
    cdef double score
    cdef np.ndarray node_scores
    cdef np.ndarray eseq
    cdef np.ndarray _translation 

cdef class SimpleSequencePath(SequencePath):
    pass 
    
cdef class SequenceBuffer:
    cdef void push(self,double cost, SequencePath path)
    cdef SequencePath pop(self)
    cdef bint empty(self)
    cdef set seen
    cdef list sortedpaths
    cdef np.ndarray kbest(self,int k)

cdef class KBestTranslations:
    cdef unicode _uinput
    cdef np.ndarray _einput,_A
    cdef int _k

## factory
cpdef Decoder(str decoder_type)
