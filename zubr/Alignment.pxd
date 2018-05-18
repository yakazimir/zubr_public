# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Classes for all alignment models 

"""

from zubr.ZubrClass cimport ZubrSerializable
from zubr.Datastructures cimport Sparse2dArray
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double log(double)
    bint isinf(double)
    bint isfinite(double)
    bint isnan(double)

cdef class AlignerBase(ZubrSerializable):
    cpdef void train(self,object config=?)

## SEQUENCE ALIGNERS
    
cdef class SequenceAligner(AlignerBase):
    cdef public object config
    cdef dict _elex,_flex
    cdef void       _train(self,np.ndarray f,np.ndarray e)
    cdef Alignment _align(self,int[:] f,int[:] e)
    cdef void _rank(self,int[:] en,np.ndarray rl,int[:] sort)
    cpdef void align_dataset(self,np.ndarray foreign,np.ndarray english,out=?)
    
cdef class WordModel(SequenceAligner):
    cdef np.ndarray table
    cdef void offline_init(self,np.ndarray foreign,np.ndarray english)
    cdef double word_prob(self,int foreign,int english,int identifier=?)
    cdef double[:,:] model_table(self)
    cdef np.ndarray model_table_np(self)
    
cdef class TreeModel(SequenceAligner):
    pass 
    
## NON DISTORITON ALIGNERS

cdef class NonDistortionModel(WordModel):
    pass

cdef class WordDistortionModel(WordModel):
    cdef np.ndarray distortion

cdef class TreeDistortionModel(TreeModel):
    cdef np.ndarray tdistortion

## aligner implementations

cdef class IBMM1(NonDistortionModel):
    pass

cdef class IBMM2(WordDistortionModel):
    pass

cdef class TreeModel2(TreeDistortionModel):
    pass

### SPARSE WORD MODELS (at some point should become main models, new development)

cdef class Sparse2dModel(Sparse2dArray):
    pass

cdef class SparseWordModel(WordModel):
    cdef Sparse2dModel sparse2d

cdef class SparseNonDistortionModel(SparseWordModel):
    pass

cdef class SparseDistortionModel(SparseWordModel):
    pass

cdef class SparseIBMM1(SparseNonDistortionModel):
    pass

cdef class SparseIBMM2(SparseDistortionModel):
    pass
    
## alignment and decoding objects

cdef class Alignment:
    cdef int slen,tlen
    cdef public double prob
    cdef np.ndarray problist,ml
    cdef list _find_best(self)

cdef class Decoding:
    cdef int tlen
    cdef public np.ndarray positions
    cdef double prob 
    cpdef unicode giza(self)


## distortion initializer

cdef inline void initialize_dist(double[:,:,:,:] dist,int maxl):
    """initialize and make uniform a distortion model"""
    cdef int K,I,i,k
    cdef double uniform

    for K in range(maxl):
        for I in range(maxl):
            for i in range(K+1):
                uniform = 1.0/len(range(I+1))
                for k in range(I+1):
                    #dist[K,I,i,k] = uniform
                    dist[K,I,k,i] = uniform
    
