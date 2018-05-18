# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementation of various kinds of neural language models using
different frameworks (currently, mostly via Dynet) 

"""
from zubr.ZubrClass cimport ZubrSerializable
import numpy as np
cimport numpy as np

### neural network modes

cdef class NeuralModel(ZubrSerializable):
    cpdef void train(self,config)

cdef class DynetModels(NeuralModel):
    cdef object dy,model,trainer

cdef class DynetFeedForward(DynetModels):
    cdef object olayer,obias
    cdef list hlayer_list,hbias_list

cdef class FeedForwardLM(DynetFeedForward):
    cdef dict flex
    cdef object context_embed

cdef class FeedForwardTranslationLM(FeedForwardLM):
    cdef int _train(self,np.ndarray f,np.ndarray e,object config) except -1
    cdef dict elex
    cdef object e_embed,len_embed,flen_embed
    cdef int epos,e_embed_dim
    cdef public int ngram

    ## general scoring functions (used for applications)
    cdef double[:] e_rep(self,int[:] x_bold)
    cdef double score(self,double[:] x,int z,int[:] z_context,int z_pos) except -1
    cdef int _rank(self,int[:] en,np.ndarray rl,int[:] sort,int gold_id) except -1

cdef class AttentionTranslationLM(FeedForwardTranslationLM):
    cdef object attention_model,attention_trainer 
    cdef list ahlayer_list,ahbias_list
