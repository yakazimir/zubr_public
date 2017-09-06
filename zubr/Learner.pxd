import numpy as np
cimport numpy as np

from zubr.ZubrClass cimport ZubrSerializable
from zubr.Features  cimport Vectorizer,FeatureObj,FeatureAnalyzer
from zubr.Dataset   cimport RankComparison,RankStorage,RankPair


cdef extern from "math.h":
    double log(double)
    
    
## base

cdef class LearnerBase(ZubrSerializable):
    pass

## BATCH vs. ONLINE LEARNERS 

cdef class BatchLearner(LearnerBase):
    pass

cdef class OnlineLearner(LearnerBase):
    cdef double online_update(self,RankPair instance,FeatureObj features,int p,int it,int size) except -1
    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks)
    cdef void finish_iteration(self,int iteration,object wdir=?)
    cdef void finish_evaluation(self)

## NEURAL LEARNERS

cdef class OnlineNeuralLearner(OnlineLearner):
    pass

cdef class OnlineFeedForward(OnlineNeuralLearner):
    pass

cdef class Layer:
    cdef double[:] forward_propogation(self,double[:] input_vector)
    cdef double[:] backward_propogation(self,double[:] input_vector)

cdef class LinearLayer(Layer):
    cdef np.ndarray w
    cdef np.ndarray b

cdef class SoftMaxLayer(Layer):
    pass 

## activations 
    
cdef class ActivationLayer(Layer):
    pass

cdef class Tanh(ActivationLayer):
    pass

cdef class Sigmoid(ActivationLayer):
    pass 

## LINEAR LEARNERS 

cdef class OnlineLinearLearner(OnlineLearner):
    cdef long _nfeatures
    cdef np.ndarray w
    cdef LinearHyper hyper
    cdef FeatureAnalyzer compute_instance(self,FeatureObj features,int lang=?)

cdef class SimpleLearner(OnlineLinearLearner):
    pass

cdef class RegularizedLearner(OnlineLinearLearner):
    cdef np.ndarray reg_time
    cdef np.ndarray learn_rates
    cdef bint normalized
    cdef int updates
    cdef void regularize(self)
        
## types of simple learners

cdef class LinearSGD(SimpleLearner):
    pass

cdef class LinearPerceptron(SimpleLearner):
    pass 

## average learners

cdef class AverageLearner(OnlineLinearLearner):
    cdef np.ndarray counts,last_update
    cdef int updates
    cdef inline void average_vector(self)

cdef class LinearAverageSGD(AverageLearner):
    pass 

cdef class LinearAveragePerceptron(AverageLearner):
    pass

cdef class LinearRegularizedSGD(RegularizedLearner):
    pass

## polyglot models

cdef class LinearPolyglot(OnlineLinearLearner):
    cdef np.ndarray lang_models
    cdef dict lang_map

cdef class PolyglotSGD(LinearPolyglot):
    pass 

cdef class AveragedSGDPolyglot(PolyglotSGD):
    pass

cdef class ModelProbs:
    cdef int num_langs
    cdef public double global_counts
    cdef np.ndarray lang_counts
    cdef public np.ndarray global_ranks,lang_ranks,global_langs
    cdef bint normalized
    cdef void reset(self)
    cpdef void normalize(self,int it=?,object wdir=?,dict lang_map=?)
    
cdef class WeightedSGDPolyglot(LinearPolyglot):
    cdef double predictions
    cdef ModelProbs model_ranks

cdef class AveragedWeightedSGDPolyglot(WeightedSGDPolyglot):
    pass 

## hyper parameter types

cdef class HyperParameterBase:
    cdef double compute_learn_rate(self,int p,int t,int dsize)

cdef class NeuralHyper(HyperParameterBase):
    pass 

cdef class LinearHyper(HyperParameterBase):
    pass

cdef class SimpleLearnRate(LinearHyper):
    cdef double regularizer
    cdef double lrate1
    cdef double lrate2

## factory

cpdef Learner(object config)

