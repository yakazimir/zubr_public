# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double log(double)
    bint isinf(double)
    bint isfinite(double)
    bint isnan(double)

cdef class Alignment:
    cdef public int slen
    cdef public int tlen
    cdef public double prob
    cdef public np.ndarray problist
    cdef public np.ndarray ml
    cdef list _find_best(self)
    cdef list _k_best(self,int k)
    cdef _k_best2(self,int k)
    cdef inline int[:] _shortest_path(self,int start, double[:,:] block)

cdef class Decoding:
    cdef public int tlen
    cdef public np.ndarray _ts_array
    cdef public double prob 
    cpdef unicode giza(self)
    
cdef class AlignerBase:
    cdef bint stops
    cdef Alignment _decode(self,int[:] s,int[:] t)
    cdef public unsigned int max_len
    cdef public str encoding
    cdef public double alambda
    cdef public double minprob
    cdef public bint ignoreoov
    cdef public np.ndarray _wc
    cdef public np.ndarray _sc
    cdef public dict source_lex
    cdef public dict target_lex
    cdef np.ndarray table
    cdef np.ndarray distortion
    cdef void _compute_freq(self,np.ndarray e)
    cdef void _compute_sfreq(self,np.ndarray f)
    cdef _rank(self,int[:] en,np.ndarray rl,int[:] sort)
    cpdef decode_dataset(self,soureced,targetd,out=?,k=?)
    cpdef tuple rank_dataset(self,object config)
    cdef void train_model1(self,np.ndarray source,np.ndarray target,
                           int maxiter,double smooth=?)
    cdef void train_model2(self,np.ndarray source,np.ndarray target,
                           int maxiter,int maxl,
                           double smooth1=?,double smooth2=?)
    
cdef class Model1(AlignerBase):
    pass

cdef class Model2(Model1):
    cpdef train(self,str path=?,config=?)

cdef class TreeModel(Model1):
    cdef np.ndarray tdistortion
    cpdef train(self,str path=?,config=?)
    cdef Alignment _tree_decode(self,int[:] s,int[:] t,int[:] treepos,int treelen)
    cpdef tuple rank_dataset(self,object config)
    cdef _rank_tree(self,int[:] en,np.ndarray rl,np.ndarray treepos,int[:] sort)
    cpdef decode_tree_dataset(self,sourced,targetd,treepos,out=?,k=?)
    cdef int train_tree_model(self,np.ndarray f, np.ndarray e,
                               np.ndarray tree_pos,int maxiter,int tmax,
                               int emax,int trainend=?,double smooth1=?,double smooth2=?) except -1
# cdef class SymmetricAligner:
#     pass


## inline functions 

cdef inline _Aligner(object config):
    """return an aligner """

    atype = ''

    if isinstance(config,basestring):
        atype = config.lower()
    else:
        atype = config.modeltype.lower()

    if atype == "ibm1":
        return Model1()
    elif atype == "ibm2":
        return Model2()
    elif atype == 'treemodel':
        return TreeModel()

    raise ('aligner model not known: %s' % atype) 

    
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

cdef inline np.ndarray _tree_dist(int maxtree,int maxelen):
    """construct the tree distortion table
    
    :param maxtree: the maximum size of tree in dataset
    :param maxelen: maximum length of e string
    :returns: an array ]
    """
    tree_dist = np.ndarray((maxtree,maxelen,maxtree),dtype='d')
    tree_dist.fill(1.0/float(maxtree))
    return tree_dist


