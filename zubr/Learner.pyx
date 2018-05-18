# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementation of Optimizers 

"""

import os
import sys
import time
import numpy as np
cimport numpy as np
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Features  cimport Vectorizer,FeatureObj,FeatureAnalyzer
from zubr.Dataset   cimport RankComparison,RankStorage,RankPair
from cython         cimport wraparound,boundscheck,cdivision

## difference types of learners

LEARNERS = {
    ## linear learners 
    "lsgd"          : LinearSGD,        
    "lasgd"         : LinearAverageSGD, 
    "lperceptron"   : LinearPerceptron,
    "laperceptron"  : LinearAveragePerceptron,
    "lrsgd"         : LinearRegularizedSGD,
    ## neural learners
    #"nsgd"         : NeuralSGD,
    ## polyglot models
    "poly_asgd"     : AveragedSGDPolyglot,
    "poly_wsgd"     : WeightedSGDPolyglot,
    "poly_awsgd"    : AveragedWeightedSGDPolyglot,
}

cdef class LearnerBase(ZubrSerializable):
    
    """Base class for learning types"""
    
    @classmethod
    def from_config(cls,config):
        """Set up a learner from a configuration 

        :param config: the learner/experiment configuration
        """
        raise NotImplementedError

    def backup(self,wdir):
        """The main method for backing up the learning models 

        :param wdir: the working directory, or place to back up 
        """
        raise NotImplementedError

    @classmethod
    def load_backup(cls,config):
        """Loads a learning model instance from backup file 

        :param config: the learner configuration
        """
        raise NotImplementedError
        
## need a base class that has score_example, at least 
    
cdef class OnlineLearner(LearnerBase):
    """Sets the scoring and update functions for online optimization algorithms"""

    cdef double online_update(self,RankPair instance,FeatureObj features,int p,int it,int size) except -1:
        """Performs an online model update

        :param current: the current scores/feature information
        :param p: the current point in training data
        :param it: the current iteration in training 
        :param size: the size of the data
        :rtype: double
        :returns: example score (if needed)
        """
        raise NotImplementedError

    cdef void finish_iteration(self,int iteration,object wdir=None):
        """Called after each training iteration
        
        -- can be used for averaging for example
        -- by default is just passed 

        :param iteration: the iteration number
        """
        pass

    cdef void finish_evaluation(self):
        """Called after running on a testing set of some kind, passed by default
        
        :rtype: None 
        """
        pass

    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks):
        """Score a particular example using current model parameters

        :param p: the point in dataset 
        :param feat: the feature representation 
        :param new_ranks: a representaiton of old/new ranks 
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls,config):
        """Set up a learner from a configuration 

        :param config: the learner/experiment configuration
        """
        raise NotImplementedError

cdef class BatchLearner(LearnerBase):
    pass


### neural learners 

cdef class OnlineNeuralLearner(OnlineLearner):
    """Online Neural network learner models"""
    pass

cdef class OnlineFeedForward(OnlineNeuralLearner):
    """An implementation of a feed forward neural network trained using an online optimization method

    Some notes: 
    """
    def __cinit__(self,int num_layers,list layers):
        """ 

        :param num_layers: the number of layers in the neural network
        :param layers: the actual network layers 
        :param non_linear: the non-linear function used 
        """
        self.num_hlayers = num_layers
        self.layers      = layers

cdef class NeuralSGD(OnlineFeedForward):
    """An implementatio of feed forward network trained with SGD"""


    @classmethod
    def from_config(cls,config):
        """Set up a learner from a configuration 

        :param config: the learner/experiment configuration
        """
        layers = []
        activation  = Activation(config)
        hsize       = config.hidden_neurons
        hlayers     = config.hidden_layers
        size_output = config.size_output

        for layer_num in range(hlayers):
            pass

        #output_layer = LinearLayer

## layer class

# (taken from http://peterroelants.github.io/posts/neural_network_implementation_part05/#Generalization-of-the-layers)
# https://github.com/andersbll/nnet

        
cdef class Layer:

    cdef double[:] forward_propogation(self,double[:] input_vector):
        """Calculate forward propogation

        :param input_vector: the input vector 
        """
        raise NotImplementedError

    cdef double[:] backward_propogation(self,double[:] input_vector):
        """Calculate backward propogation

        :param input_vector: the input vector 
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls,config):
        """Set up layers from configuration

        :param configuration: the main configuration 
        """
        raise NotImplementedError

cdef class LinearLayer(Layer):
    """A simple linear layer class"""

    def __init__(self,weights,bias):
        """Initializes a linear layer instance 

        :param weights: the layer parameters 
        :param bias: the bias vector 
        """
        self.b = weights
        self.w = bias


    cdef double[:] forward_propogation(self,double[:] input_vector):
        """Calculate forward propogation

        :param input_vector: the input vector 
        """
        cdef double[:] b = self.bias
        cdef double[:,:] w = self.weights
        self.last_input = input_vector

        return np.add(np.dot(input_vector,w),b)

    cdef double[:] backward_propogation(self,double[:] input_vector):
        """Calculate backward propogation

        :param input_vector: the input vector 
        """
        cdef double[:] b = self.bias
        cdef double[:,:] w = self.weights
        
        return np.dot(input_vector,w.T)
        

    @classmethod
    def from_vals(cls,input_s,output_s):
        np.random.seed(0)
        weights = np.random.randn(input_s,output_s) * 0.1
        bias = np.zeros(output_s,dtype='d')

        return cls(weights,bias)


## activations

ACTIVE = {
    "tanh"    : Tanh,
    "sigmoid" : Sigmoid,
}

cpdef Activation(config):
    """Activation factor"""
    act = config.non_linear
    if act not in ACTIVE:
        raise ValueError('Unknown activation function..')
    return ACTIVE[act]()


cdef class ActivationLayer(Layer):
    """Non-linear activation layer class"""
    pass 


cdef class Sigmoid(ActivationLayer):

    cdef double[:] forward_propogation(self,double[:] input_vector):
        """Calculate forward propogation

        :param input_vector: the input vector 
        """
        return np.divide(1.0,np.add(1.0,np.exp(np.negative(input_vector))))

    cdef double[:] backward_propogation(self,double[:] input_vector):
        """Calculate forward propogation

        :param input_vector: the input vector 
        """
        cdef double[:] s = np.divide(1.0,np.add(1.0,np.exp(np.negative(input_vector))))
        return np.multiply(s,np.minus(1,s))

cdef class Tanh(ActivationLayer):

    cdef double[:] forward_propogation(self,double[:] input_vector):
        """Calculate forward propogation

        :param input_vector: the input vector 
        """
        return np.tanh(input_vector)

    cdef double[:] backward_propogation(self,double[:] input_vector):
        """Calculate forward propogation

        :param input_vector: the input vector 
        """
        cdef double[:] e = np.exp(np.multiply(2,input_vector))
        return np.divide(np.minus(e,1),np.add(e,1))
    
    
cdef class SoftMaxLayer(Layer):

    """Softmax layer class"""
    
    cdef double[:] forward_propogation(self,double[:] input_vector):
        """Calculate forward propogation

        :param input_vector: the input vector 
        """
        cdef double[:] e,f
        e = np.exp(np.minus(input_vector,np.amax(input_vector,axis=1,keepdims=True)))
        f = np.divide(e,np.sum(e,axis=1,keepdims=True))
        return f

    cdef double[:] backward_propogation(self,double[:] input_vector):
        """Calculate backward propogation

        :param input_vector: the input vector 
        """
        raise NotImplementedError


## linear learners

cdef class OnlineLinearLearner(OnlineLearner):

    """
    Online learner for linear models 
    """
    
    property  nfeatures:
        """The number of features or parameters in the model"""

        def __get__(self):
            """Returns the current number of features 

            :rtype: lone 
            """
            return <long>self._nfeatures

    cdef FeatureAnalyzer compute_instance(self,FeatureObj features,int lang=-1):
        """Runs the model on a particular training instance 

        -- e.g., might be used to compute current likelihood, etc..

        :param features: the current features 
        """
        raise NotImplementedError


cdef class SimpleLearner(OnlineLinearLearner):
    """Simple linear learners that just have one parameter set"""
    
    def __cinit__(self,long num_features,np.ndarray w,LinearHyper hyper):
        """Initializes a simple learner model 

        :param num_features: the number of features in model
        """
        if num_features <= 0:
            raise ValueError('Model must have feature size >= 1')

        self._nfeatures = num_features
        self.w = w
        self.hyper = <LinearHyper>hyper

    @classmethod
    def from_config(cls,config):
        """Set up a learner from a configuration 

        :param config: the learner/experiment configuration
        """
        feat_num = config.num_features
        hypers   = HyperParameters(config)
        w = np.zeros((feat_num,),dtype='d')
        return cls(feat_num,w,hypers)

        ## backup protocol
        
    def backup(self,wdir):
        """Write the instance to file for later use 

        :param wdir: the working directory 
        """
        stime = time.time()
        ldir = wdir #os.path.join(wdir,"learner")

        ### back up the hyper parameters object (use pickle)
        self.hyper.backup(wdir)

        ## back up the weight components and 
        fout = os.path.join(ldir,"learner_components")
        np.savez_compressed(fout,self.w,np.array([self._nfeatures]))

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load a learning instance from file 

        :param config: the global configuration 
        """
        stime = time.time()
        ldir = config.dir #os.path.join(config.dir,"learner")

        ## the component
        rpath = os.path.join(ldir,"learner_components.npz")
        archive = np.load(rpath)
        w = archive["arr_0"]
        num_features = archive["arr_1"][0]

        ## hyper parameter class
        hclass = HyperParameters(config)
        hypers = hclass.load_backup(config)

        instance = cls(num_features,w,hypers)
        instance.logger.info('Loaded in %s seconds' % str(time.time()-stime))
        return instance

cdef class RegularizedLearner(OnlineLinearLearner):
    """Learners that use regularization, which is computed using time stamp trick"""

    def __cinit__(self,long num_features,
                      np.ndarray w,
                      np.ndarray reg_time,
                      np.ndarray learn_rates,
                      int updates,
                      LinearHyper hyper,
                      ):
        """Create a regularized learner instance 

        :param num_features: the number of features in the model 
        :param w: the weight vector 
        :param reg_time: a record of when items were last regularized 
        :param updates: the number of total updates
        :param hyper: hyper parameters
        """
        if num_features <= 0:
            raise ValueError('Model must have feature size >= 1')
        
        self._nfeatures = num_features
        self.w = w
        self.reg_time = reg_time
        self.updates = updates
        self.hyper = hyper
        self.learn_rates = learn_rates
        
    @classmethod
    def from_config(cls,config):
        """Creates an instance from a configuration 

        :param config: the learner or experiment configuration
        """
        feat_num = config.num_features
        hypers = HyperParameters(config)
        w = np.zeros((feat_num,),dtype='d')
        reg = np.zeros((feat_num,),dtype=np.int32)
        lr = np.zeros((0,),dtype=np.int32)
        
        return cls(feat_num,w,reg,lr,0,hypers)

    cdef void regularize(self):
        """Called after an iteration, make sure all features are updated in terms of 
        regularization 

        :rtype: None
        """
        cdef int num_updates = self.updates
        cdef int[:] reg_time = self.reg_time
        cdef double[:] w = self.w
        cdef double[:] learn_rates = self.learn_rates
        cdef long num_feat = self._nfeatures
        cdef long i
        cdef int j
        cdef int last_time,gap
        cdef LinearHyper hypers = <LinearHyper>self.hyper
        cdef double reg = hypers.lambda_term

        for i in range(num_feat):
            last_time = reg_time[i]
            if w[i] == 0: continue 

            ## catch up the unregularized items
            for j in range(last_time,num_updates):
                w[i] += learn_rates[j]*(0.0 - (reg*w[i]))

            reg_time[i] = num_updates

    cdef void finish_iteration(self,int iteration,object wdir=None):
        """Called after each training iteration
        
        -- can be used for averaging for example
        -- by default is just passed 

        :param iteration: the iteration number
        """
        s_time = time.time()
        self.logger.info('Updating regularization...')
        self.regularize()
        self.logger.info('Updated in %s seconds' % str(time.time()-s_time))


    def backup(self,wdir):
        """Write the average learner instance to file 

        :param wdir: the working directory 
        """
        stime = time.time()
        ldir = wdir #os.path.join(wdir,"learner")

        ### back up the hyper parameters object (use pickle)
        #self.hyper.dump(os.path.join(ldir,"hyper_params"))
        self.hyper.backup(wdir)
        
        ## back up the
        fout = os.path.join(ldir,"learner_components")
        np.savez_compressed(fout,self.w,self.reg_time,self.learn_rates,
                                np.array([self._nfeatures],np.array([self.updates])))

        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load a learner from backup file 

        :param config: the main experiment configuration 
        """
        stime = time.time()
        ldir = config.dir #os.path.join(config.dir,"learner")

        rpath = os.path.join(ldir,"learner_components.npz")
        archive = np.load(rpath)
        w = archive["arr_0"]
        rtime = archive["arr_1"]
        lrates = archive["arr_2"]
        num_features = archive["arr_3"][0]
        updates = archive["arr_4"][0]

        ## hyper parameter class
        hclass = HyperParameters(config)
        hypers = hclass.load_backup(config)
        
        instance = cls(num_features,w,rtime,lrates,updates,hypers)
        instance.logger.info('Loaded in %s seconds' % str(time.time()-stime))
        return instance
        
        
cdef class AverageLearner(OnlineLinearLearner):
    """Learners that involve averaging of some kind, maintain two parameter sets"""

    def __cinit__(self,long num_features,
                      np.ndarray w,np.ndarray a,
                      np.ndarray last_update,
                      updates,
                      LinearHyper hyper
                      ):
        """Initializes a simple learner model 

        :param num_features: the number of features in model
        """
        if num_features <= 0:
            raise ValueError('Model must have feature size >= 1')

        self._nfeatures = num_features
        self.w = w
        self.counts = a
        self.last_update = last_update
        self.hyper  = hyper
        self.updates = updates
        
    @classmethod
    def from_config(cls,config):
        """Set up a learner from a configuration 

        :param config: the learner/experiment configuration
        """
        feat_num = config.num_features
        hypers   = HyperParameters(config)
        w  = np.zeros((feat_num,),dtype='d')
        a  = np.zeros((feat_num,),dtype='d')
        lu = np.zeros((feat_num,),dtype=np.int32)
        
        return cls(feat_num,w,a,lu,0,hypers)
    
    cdef inline void average_vector(self):
        """Update the average weight vector counts, usually performed after 
        iteration

        :rtype: None
        """
        cdef double[:] weights = self.w
        cdef double[:] counts  = self.counts
        cdef int[:] last_update = self.last_update
        cdef int updates = self.updates
        cdef long i,num_weights = weights.shape[0]
        cdef int gap
        
        for i in range(num_weights):
            gap = updates - last_update[i]
            counts[i] += weights[i]*gap
            last_update[i] = updates

    cdef void finish_iteration(self,int iteration,object wdir=None):
        """Called after each training iteration
        
        -- can be used for averaging for example
        -- by default is just passed 

        :param iteration: the iteration number
        """
        self.logger.info('Updating average vector counts...')
        st = time.time()
        self.average_vector()
        self.logger.info('Finished counting in %s seconds' % str(time.time()-st))

    def backup(self,wdir):
        """Write the average learner instance to file 

        :param wdir: the working directory 
        """
        stime = time.time()
        ldir = wdir #os.path.join(wdir,"learner")

        ### back up the hyper parameters object (use pickle)
        #self.hyper.dump(os.path.join(ldir,"hyper_params"))
        self.hyper.backup(wdir)

        ## back up the
        fout = os.path.join(ldir,"learner_components")
        np.savez_compressed(fout,self.w,self.counts,self.last_update,
                                np.array([self._nfeatures]),np.array([self.updates]))

        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load a learner from backup file 

        :param config: the main experiment configuration 
        """
        stime = time.time()
        ldir = config.dir #os.path.join(config.dir,"learner")

        rpath = os.path.join(ldir,"learner_components.npz")
        archive = np.load(rpath)
        w = archive["arr_0"]
        a = archive["arr_1"]
        last_updates = archive["arr_2"]
        num_features = archive["arr_3"][0]
        updates = archive["arr_4"][0]

        ## hyper parameter class
        hclass = HyperParameters(config)
        hypers = hclass.load_backup(config)
        
        instance = cls(num_features,w,a,last_updates,updates,hypers)
        instance.logger.info('Loaded in %s seconds' % str(time.time()-stime))
        return instance

### each class below just implements a ``compute_instance``
## function and an ``online_update`` function

        
cdef class LinearSGD(SimpleLearner):
    """
    Vanilla stochastic gradient ascent/descent optimization 
    the using a simple LCL objective
    """

    cdef FeatureAnalyzer compute_instance(self,FeatureObj features,int lang=-1):
        """Runs the model on a particular training instance 

        -- e.g., might be used to compute current likelihood, etc..

        :param features: the current features 
        """
        cdef double[:] weights = self.w
        return <FeatureAnalyzer>train_prob(features,weights)

    cdef double online_update(self,RankPair instance,FeatureObj features,int p,int it,int size) except -1:
        """Performs a standard sgd update using LCL objective

        :param features: the feature represetation for given example
        :param p: the point in training data 
        :param it: the current iteration 
        :param size: the size of the training data
        """
        cdef FeatureAnalyzer scores = self.compute_instance(features)
        cdef LinearHyper hypers = <LinearHyper>self.hyper
        cdef double lrate
        cdef double reg = hypers.lambda_term
        cdef double[:] weights = self.w
        cdef long num_feat = self._nfeatures
        
        lrate = hypers.compute_learn_rate(p,it,size)
        try: 
            if reg == 0.0: 
                sgd_update(scores,weights,lrate,reg)
            else:
                reg_sgd_update(scores,weights,num_feat,lrate,reg)
        except Exception,e:
            self.logger.error(e,exc_info=True)
            sys.exit('Exited, look at log')
            
        return <double>scores.likelihood

    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks):
        """Score a particular example using current model parameters

        :param p: the point in dataset 
        :param feat: the feature representation 
        :param new_ranks: a representaiton of old/new ranks 
        :returns: the probability of the correct example (if specified)
        """
        cdef double[:] weights = self.w
        cdef FeatureAnalyzer analysis
        
        analysis = score(p,feat,new_ranks,weights)
        return analysis.likelihood

    def __reduce__(self):
        return LinearSGD,(self._nfeatures,self.w,self.hyper)

        
cdef class LinearPerceptron(SimpleLearner):
    """
    Vanilla perceptron learner, i.e., optimizes using the perceptron update rule
    """

    cdef double online_update(self,RankPair instance,FeatureObj features,int p,int it,int size) except -1:
        """Score a particular example for perceptron learning

        :param features: the current features 
        :param p: the point in the training data 
        :param it: the current iteration 
        :param size: the size of the training data
        """
        cdef double[:] weights = self.w
        cdef FeatureAnalyzer analysis = train_prob(features,weights,norm=False)
        cdef double[:] scores = analysis.probs
        cdef int best    = np.argmax(scores)
        cdef double ssum = np.sum(scores)

        if best != 0:
            perceptron_update(analysis,best,weights)
            
        elif best == 0 and ssum == 0:
            perceptron_update(analysis,1,weights)
            
        ## return 1.0 as likelihood
        return 1.0

    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks):
        """Score a particular example using current model parameters

        :param p: the point in dataset 
        :param feat: the feature representation 
        :param new_ranks: a representaiton of old/new ranks 
        :returns: the probability of the correct example (if specified)
        """
        cdef double[:] weights = self.w
        cdef FeatureAnalyzer analysis
        
        analysis = score(p,feat,new_ranks,weights,norm=False)
        return 1.0
    
    def __reduce__(self):
        return LinearPerceptron,(self._nfeatures,self.w,self.hyper)


cdef class LinearAverageSGD(AverageLearner):
    """SGD where averaging is done during updates"""

    ## this function is not actually needed 
    cdef FeatureAnalyzer compute_instance(self,FeatureObj features,int lang=-1):
        """Runs the model on a particular training instance 

        -- e.g., might be used to compute current likelihood, etc..

        :param features: the current features 
        """
        cdef double[:] weights = self.w
        return <FeatureAnalyzer>train_prob(features,weights)

    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks):
        """Score a particular example using current model parameters

        :param p: the point in dataset 
        :param feat: the feature representation 
        :param new_ranks: a representaiton of old/new ranks 
        :returns: the probability of the correct example (if specified)
        """
        cdef double[:] counts = self.counts
        cdef int num_updates = self.updates
        cdef FeatureAnalyzer analysis
        
        analysis = ascore(p,feat,new_ranks,counts,num_updates)
        return analysis.likelihood

    cdef double online_update(self,RankPair instance,FeatureObj features,int p,int it,int size) except -1:
        """Performs a standard sgd update using LCL objective

        :param features: the feature represetation for given example
        :param p: the point in training data 
        :param it: the current iteration 
        :param size: the size of the training data
        """
        cdef double[:] weights = self.w
        cdef FeatureAnalyzer scores = train_prob(features,weights)
        cdef LinearHyper hypers = <LinearHyper>self.hyper
        cdef double lrate
        cdef double reg = hypers.lambda_term
        cdef double[:] counts = self.counts
        cdef int[:] last_update = self.last_update

        self.updates += 1
        
        lrate = hypers.compute_learn_rate(p,it,size)
        asgd_update(scores,weights,counts,last_update,self.updates,it,lrate,reg)
        
        return <double>scores.likelihood
    
    def __reduce__(self):
        return LinearAverageSGD,(self._nfeatures,self.w,self.counts,self.last_update,
                                     self.updates,self.hyper)

cdef class LinearAveragePerceptron(AverageLearner):
    """Average perceptron implementation"""


    cdef double online_update(self,RankPair instance,FeatureObj features,int p,int it, int size) except -1:
        """Update rule for the average perceptron algorithm 

        :param features: the current features to make updates on 
        :param p: the point in the training data 
        :param it: the current iteration 
        :param size: the size of the training data 
        """
        cdef double[:] weights = self.w
        cdef double[:] counts  = self.counts
        cdef int[:] last_update = self.last_update
        cdef FeatureAnalyzer analysis = train_prob(features,weights,norm=False)
        cdef double[:] scores = analysis.probs
        cdef int best    = np.argmax(scores)
        cdef double ssum = np.sum(scores)

        ## increment number of updates 
        self.updates += 1
        
        if best != 0:
            aperceptron_update(analysis,best,weights,counts,last_update,self.updates)

        elif best == 0 and ssum == 0:
            aperceptron_update(analysis,1,weights,counts,last_update,self.updates)

        return 1.0

    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks):
        """Score a particular example using current (averaged) model parameters

        -- Note: assumes average counts are updated, should be done if it 
        was trained using a standard online optimizer loop.

        :param p: the point in dataset 
        :param feat: the feature representation 
        :param new_ranks: a representaiton of old/new ranks 
        :returns: the probability of the correct example (if specified)
        """
        cdef double[:] counts = self.counts
        cdef int num_updates = self.updates

        ascore(p,feat,new_ranks,counts,num_updates,norm=False)
        return 1.0

    def __reduce__(self):
        ## Pickle implementation 
        return LinearAveragePerceptron,(self._nfeatures,self.w,self.counts,self.last_update,
                                            self.updates,self.hyper)
            
cpdef Learner(object config):
    """Factor method, returns the type of learner to use
    
    :param config: the learner configuration 
    :returns: the learner class
    :raises: ValueError 
    """
    learner_type = config.learner.lower()
    if learner_type not in LEARNERS:
        raise ValueError('Unknown learner type: %s' % learner_type)
    
    lclass = LEARNERS[learner_type]
    
    return lclass.from_config(config)

## HYPER PARAMETERS

cdef class HyperParameterBase:
    """Base class for storing and computing hyperparameters"""
    cdef double compute_learn_rate(self,int p,int t,int dsize):
        """Compute the learning rate at a given example 

        :param p: the position in the current iteration
        :param epochs: the total number of epochs
        :param dsize: the dataset size 
        :returns: learning rate
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls,config):
        """Load hyper parameters from config

        :param config: the experiment configuration
        """
        raise NotImplementedError

cdef class NeuralHyper(HyperParameterBase):
    """Base class for general hyper parameters"""
    pass

cdef class LinearHyper(HyperParameterBase):
    """Base class for linear model hyper parameters"""
    pass

cdef class SimpleLearnRate(LinearHyper):

    def __init__(self,lrate1,lrate2,rlambda):
        """Initializes a hyper parameter instance using a config

        :param config: the overall configuration
        """
        self.lrate1      = lrate1
        self.lrate2      = lrate2
        self.regularizer = rlambda

    cdef double compute_learn_rate(self,int p,int t,int dsize):
        """Compute the learning rate using epoch and iteration information and 
        two learning rates 

        :param p: the position in the training data currently 
        :param t: the current epoch, or iteration in algorithm 
        :para dsize: the dataset size 
        :rtype: double
        :returns: the learning rate
        """
        cdef double lrate1 = self.lrate1
        cdef double lrate2 = self.lrate2
        cdef double z = p+(t*dsize)
        return <double>lrate1/(1.0+(lrate2*z))

    @classmethod
    def from_config(cls,config):
        """Load from configuration 

        :param config: the overall configuration 
        """
        return cls(config.lrate1,config.lrate2,config.rlambda)
                
    property lambda_term:
        """The regularizer term and hyper parameter"""

        def __get__(self):
            """Returns the current regulairzer term

            :rtype: double
            """
            return <double>self.regularizer

        def __set__(self,double new_term):
            """Changes or sets the value of the new lambda term

            :param new_term: the new term
            :rtype: None 
            """
            self.regularizer = new_term

    def __reduce__(self):
        return SimpleLearnRate,(self.lrate1,self.lrate2,self.regularizer)

    ## backup protocol

    def backup(self,wdir):
        """Back up the item to file

        :param wdir: the working directory, place to dump items
        """
        hpath = os.path.join(wdir,"hypers")
        np.savez_compressed(hpath,np.array([self.lrate1,self.lrate2,self.regularizer],dtype='d'))

    @classmethod
    def load_backup(cls,config):
        """Load item from file

        :param config: the main configuration 
        """
        hpath = os.path.join(config.dir,"hypers.npz")
        archive = np.load(hpath)
        values = archive["arr_0"]
        return  cls(values[0],values[1],values[2])

## regularized sgd

cdef class LinearRegularizedSGD(RegularizedLearner):
    """Regularized variant of SGD"""

    cdef double online_update(self,RankPair instance,FeatureObj features,int p,int it,int size) except -1:
        """Performs an online sgd update with regularization

        :param features: the current features
        :param p: the current point in the training data 
        :param it: the current iteration 
        :param size: the size of the data 
        """
        cdef double[:] weights = self.w
        cdef FeatureAnalyzer scores = train_prob(features,weights)
        cdef LinearHyper hypers = <LinearHyper>self.hyper
        cdef double lrate
        cdef double reg = hypers.lambda_term
        cdef int[:] reg_time = self.reg_time
        
        self.updates += 1
        lrate = hypers.compute_learn_rate(p,it,size)
        
        self.learn_rates = np.append(self.learn_rates,[lrate])
        rsgd_update(scores,weights,reg_time,
                            self.updates,self.learn_rates,lrate,reg)

        return <double>scores.likelihood

    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks):
        """Score a particular example using current model parameters

        :param p: the point in dataset 
        :param feat: the feature representation 
        :param new_ranks: a representaiton of old/new ranks 
        :returns: the probability of the correct example (if specified)
        """
        cdef double[:] weights = self.w
        cdef FeatureAnalyzer analysis
        
        analysis = score(p,feat,new_ranks,weights)
        return analysis.likelihood

    def __reduce__(self):
        ## pickle implementation
        return LinearRegularizedSGD,(self._nfeatures,
                                         self.w,self.reg_time,
                                         self.learn_rates,self.updates,self.hyper)

### Polyglot linear models

## model 1 : averages weights after each iteration, normal scoring/prediction 
## model 2 : keeps models separate, does weighted voting when scoring
## model 3 : averages after iteration, does weighted voting when scoring

cdef class LinearPolyglot(OnlineLinearLearner):
    """A linear model that has multiple local models for different languages

    When initializing from configuration, note that this model requires a file
    {name_of_data}.language, which specifies the training data language types.
    """

    def __cinit__(self,long num_features,np.ndarray w,np.ndarray lang_models,LinearHyper hyper,lang_map):
        """Initializes a LinearPolyglot model 

        :param num_features: the number of features in the model 
        :param w: the main weight vector for the global model 
        :param lang_models: the individual models per each language 
        :param hyper: the global hyper parameters 
        :param lang_map: the language lookup map 
        """
        self._nfeatures = num_features
        self.w = w
        self.lang_models = lang_models
        self.hyper = <LinearHyper>hyper
        self.lang_map = lang_map

    def backup(self,wdir):
        """Write the instance to file for later use 

        :param wdir: the working directory 
        """
        stime = time.time()
        ldir = wdir

        ### back up the hyper parameters object (use pickle)
        self.hyper.backup(wdir)
        
        ## back up the weight components and 
        fout = os.path.join(ldir,"learner_components")
        np.savez_compressed(fout,self.w,self.lang_models,self.lang_map,
                                np.array([self._nfeatures]))

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

        
    def load_backup(cls,config):
        """Load a learner from backup file 

        :param config: the main experiment configuration 
        """
        stime = time.time()
        ldir = config.dir

        rpath = os.path.join(ldir,"learner_components.npz")
        archive = np.load(rpath)
        w = archive["arr_0"]
        lang_models = archive["arr_1"]
        lang_map = archive["arr_2"].item()
        num_features = archive["arr_3"][0]

        ## hyper parameter class
        hclass = HyperParameters(config)
        hypers = hclass.load_backup(config)

        instance = cls(num_features,w,lang_models,hypers,lang_map)
        instance.logger.info('Loaded in %s seconds' % str(time.time()-stime))
        return instance

    @classmethod
    def from_config(cls,config):
        """Load a linear SGD instance from configuration

        :param config: the main configuration 
        """
        feat_num = config.num_features

        ## standard stuff 
        hypers = HyperParameters(config)
        w = np.zeros((feat_num,),dtype='d')

        ## individual language models and lang lookup
        lang_map = find_lang_list(config)
        lang_models = np.zeros((len(lang_map),feat_num),dtype='d')
        
        return cls(feat_num,w,lang_models,hypers,lang_map)

    cdef FeatureAnalyzer compute_instance(self,FeatureObj features,int lang=-1):
        """Runs the model on a particular training instance 

        -- e.g., might be used to compute current likelihood, etc..

        :param features: the current features 
        """
        cdef double[:] weights = self.w
        cdef double[:,:] lang_weights = self.lang_models
        if lang == -1: 
            return <FeatureAnalyzer>train_prob(features,weights)
        return <FeatureAnalyzer>train_prob(features,lang_weights[lang])

cdef class PolyglotSGD(LinearPolyglot):
    """A Linear polyglot model that trains using SGD"""

    cdef double online_update(self,RankPair instance,FeatureObj features, int p,int it,int size) except -1:
        """Performs an online sgd update using LCL objective, one for the global model, and a second
        for the particular language 

        :param features: the feature representation for the given example 
        :param p: the point in the training data 
        :param it: the current iteration 
        :param size: the size of the training data 
        """
        cdef FeatureAnalyzer loc_scores,gen_scores = self.compute_instance(features)
        
        ## hyper parameters 
        cdef LinearHyper hypers = <LinearHyper>self.hyper
        cdef double lrate = hypers.compute_learn_rate(p,it,size)

        ## model weights
        cdef double[:] lang_model,weights = self.w
        cdef double[:,:] local_models = self.lang_models
        cdef long num_feat = self._nfeatures

        ## language information 
        cdef object lang = instance.lang
        cdef dict lmap = self.lang_map
        cdef int model_id = -1 if lang is None else lmap.get(lang,-1)

        ## update general model
        try: 
            sgd_update(gen_scores,weights,lrate,0.0)
        except Exception,e:
            self.logger.error(e,exc_info=True)
            sys.exit('Exited while doing main model updates, see log')

        ## update local language model
        if model_id == -1:
            self.logger.warning('Unknown model id: %s' % lang)
        else:
            ## update on local model
            try:
                loc_scores = self.compute_instance(features,lang=model_id)
                lang_model = local_models[model_id]
                sgd_update(loc_scores,lang_model,lrate,0.0)
            except:
                self.logger.error(e,exc_info=True)
                sys.exit('Exited while doing local model updates, see log')

        ## return the main model scores
        return <double>gen_scores.likelihood

    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks):
        """Score a particular example using current model parameters

        :param p: the point in dataset 
        :param feat: the feature representation 
        :param new_ranks: a representaiton of old/new ranks 
        :returns: the probability of the correct example (if specified)
        """
        cdef double[:] weights = self.w
        cdef FeatureAnalyzer analysis
        
        analysis = score(p,feat,new_ranks,weights)
        return analysis.likelihood

cdef class AveragedSGDPolyglot(PolyglotSGD):
    """This model is a polyglot model that trains a global ``polyglot`` model over 
    the entire dataset, and local models for each language. After each iteration, 
    it averages the global weights overall all models. It scores using this averaged 
    global model in the end.
    """

    cdef void finish_iteration(self,int iteration,object wdir=None):
        """Average the main weight vector after a given iteration 

        :param iteration: the current iteration just finished
        """
        cdef double[:] weights = self.w
        cdef double[:,:] local_models = self.lang_models
        cdef long num_feat = self._nfeatures
        
        self.logger.info('Averaging the main model...')
        poly_average(num_feat,weights,local_models)

    def __reduce__(self):
        ## pickle implementation
        return AveragedSGDPolyglot,(self._nfeatures,
                                    self.w,self.lang_models,
                                    self.hyper,self.lang_map)

## helped class for weighted polyglot models

cdef class ModelProbs:
    """Class to keep track of model probabilities and p(model | language) probabilities"""

    def __init__(self,num_langs):
        """Create a model probs instance
        
        :param num_langs: the number of languages involved
        """
        ## predictions
        self.num_langs = num_langs
        self.global_counts = 0.0
        self.lang_counts = np.zeros((num_langs,),dtype='d')

        ## rank scores 
        self.global_ranks = np.zeros((num_langs+1,),dtype='d')
        self.lang_ranks   = np.zeros((num_langs,num_langs),dtype='d')
        self.global_langs = np.zeros((num_langs,),dtype='d')
        
        ## averaged
        self.normalized = False
        
    cpdef void  normalize(self,int it=-1,object wdir=None, dict lang_map={}):
        """Normalizes the counts to probabilities, using softmax normalization with 
        a heuristically set temperature parameter of 0.1

        :param wdir: the working directory 
        :param it: the number of current iterations 
        :param lang_map: the map of language values 
        """
        cdef np.ndarray[ndim=1,dtype=np.double_t] global_ranks = self.global_ranks
        cdef np.ndarray[ndim=1,dtype=np.double_t] lang_counts = self.lang_counts
        cdef np.ndarray[ndim=2,dtype=np.double_t] lang_ranks = self.lang_ranks
        cdef np.ndarray[ndim=1,dtype=np.double_t] global_langs = self.global_langs
        cdef int lang,other_lang,num_langs = lang_counts.shape[0]
        cdef double denominator,loc_score

        ## global normalization
        
        if self.global_counts > 0.0:
            global_ranks[0] = np.exp((global_ranks[0]/self.global_counts)/0.01)
            denominator = 0.0+global_ranks[0]

            ## find denominator
            for lang in range(num_langs):
                print_type = 'w'
                global_ranks[lang+1] = np.exp((global_ranks[lang+1]/self.global_counts)/0.01)
                denominator  += global_ranks[lang+1]

            ## normalize
            global_ranks[0] = global_ranks[0]/denominator
            for lang in range(num_langs):
                global_ranks[lang+1] = global_ranks[lang+1]/denominator

        ## print?
        if wdir:
            print_model_scores(it,wdir,
                                global_ranks,
                                None,
                                "global",
                                "w",
                                lang_map,
                                sglobal=True)

        denominator = 0.0
        ## language specific distribution 
        for lang in range(num_langs):
            print_type = 'a'
            lang_count = lang_counts[lang]
            
            if lang_count != 0.0: 
                loc_score = np.exp((global_langs[lang]/lang_count)/0.01)
                denominator = 0.0 + loc_score
                global_langs[lang] = loc_score

                ## calculate denominator
                for other_lang in range(num_langs):
                    loc_score = np.exp((lang_ranks[lang][other_lang]/lang_count)/0.01)
                    denominator += loc_score
                    lang_ranks[lang][other_lang] = loc_score 

                ## normalize counts
                global_langs[lang] = global_langs[lang]/denominator
                for other_lang in range(num_langs):
                    lang_ranks[lang][other_lang] = lang_ranks[lang][other_lang]/denominator

            ## print out the result? 
            if wdir:
                print_model_scores(it,wdir,
                                       lang_ranks[lang],
                                       global_langs,
                                       lang,
                                       print_type,
                                       lang_map)


    cdef void reset(self):
        """Reset the model counts

        :rtype: None 
        """
        cdef double[:] lang_counts = self.lang_counts
        cdef double[:] global_ranks = self.global_ranks
        cdef double[:] global_langs = self.global_langs
        cdef double[:,:] lang_ranks = self.lang_ranks

        ## puts everyhing back to zero in order to start counting again
        self.global_counts = 0.0
        lang_counts[:]     = 0.0
        global_ranks[:]    = 0.0
        lang_ranks[:,:]    = 0.0
        global_langs[:]    = 0.0

        ### set off the normalized switch
        self.normalized    = False

    def __reduce__(self):
        ModelProbs,(self.num_langs,)

        
cdef class WeightedSGDPolyglot(LinearPolyglot):
    """These are polyglot models that keep track of model accuracy, which are 
    used to weight model averaging, and potentially other operations (e.g., model
    voting) 
    """
    def __cinit__(self,long num_features,np.ndarray w,np.ndarray lang_models,LinearHyper hyper,lang_map):
        """Initializes a LinearPolyglot model 

        :param num_features: the number of features in the model 
        :param w: the main weight vector for the global model 
        :param lang_models: the individual models per each language 
        :param hyper: the global hyper parameters 
        :param lang_map: the language lookup map 
        """
        self._nfeatures = num_features
        self.w = w
        self.lang_models = lang_models
        self.hyper = <LinearHyper>hyper
        self.lang_map = lang_map

        ## datastructures for keeping track of individual model performance
        num_models = lang_models.shape[0]
        self.model_ranks = ModelProbs(num_models)

    cdef FeatureAnalyzer compute_instance(self,FeatureObj features,int lang=-1):
        """Runs the model on a particular training instance 

        -- e.g., might be used to compute current likelihood, etc..

        :param features: the current features 
        """
        cdef double[:] weights = self.w
        cdef double[:,:] lang_weights = self.lang_models
        
        if lang == -1: 
            return <FeatureAnalyzer>weighted_train_prob(features,weights)
        return <FeatureAnalyzer>weighted_train_prob(features,lang_weights[lang])

    cdef double score_example(self,RankPair instance,int p,FeatureObj feat,RankComparison new_ranks):
        """Score a particular example using current model parameters

        :param p: the point in dataset 
        :param feat: the feature representation 
        :param new_ranks: a representaiton of old/new ranks 
        :returns: the probability of the correct example (if specified)
        """
        cdef double[:] weights = self.w
        cdef FeatureAnalyzer analysis

        ## language information
        cdef object lang = instance.lang
        cdef dict lmap = self.lang_map
        cdef int model_id = -1 if lang is None else lmap.get(lang.strip(),-1)

        ## other models
        cdef double[:,:] local_models = self.lang_models

        ## model probabilities
        cdef ModelProbs model_scores = self.model_ranks
        cdef double[:,:] lang_ranks = model_scores.lang_ranks
        cdef double likelihood = 0.0
        
        ## unknown languages 
        if model_id == -1:
            analysis = score(p,feat,new_ranks,weights)
            self.logger.warning('Target language unknown, not voting...')
            likelihood = analysis.likelihood

        ## known languages
        else:

            try: 
                likelihood = weighted_scoring(p,feat,
                                                new_ranks,
                                                weights,
                                                model_id,
                                                local_models,
                                                lang_ranks[model_id])
                
            except Exception,e:
                self.logger.info(e,exc_info=True)

        #return analysis.likelihood
        #return analysis
        return likelihood

    cdef double online_update(self,RankPair instance,FeatureObj features,int p,int it,int size) except -1:
        """Performs an online update on the global model and local language model, which in addition
        computing the accuracy of all other models on correctly predicting the incoming rank list features


        :param instance: the training instance, which contains info about language type 
        :param features: the incoming features 
        :param p: the point in the training data (used for computing the hyper parameters) 
        :param it: the current iteration (again, for hyper parameters) 
        :param size: the size of the training data 
        """
        cdef FeatureAnalyzer other_scores,loc_scores,gen_scores = self.compute_instance(features)
        cdef double reranked_loc
        
        ## hyper parameters 
        cdef LinearHyper hypers = <LinearHyper>self.hyper
        cdef double lrate = hypers.compute_learn_rate(p,it,size)

        ## model weights
        cdef double[:] lang_model,weights = self.w
        cdef double[:,:] local_models = self.lang_models
        cdef long num_feat = self._nfeatures

        ## language information 
        cdef object lang = instance.lang
        cdef dict lmap = self.lang_map
        cdef int model_id = -1 if lang is None else lmap.get(lang.strip(),-1)
        cdef int i,num_langs = len(lmap)

        ## prediction and score information
        cdef ModelProbs model_ranks = self.model_ranks
        cdef double[:] global_ranks = model_ranks.global_ranks
        cdef double[:,:] lang_ranks = model_ranks.lang_ranks
        cdef double[:] lang_counts  = model_ranks.lang_counts
        cdef double[:] global_langs = model_ranks.global_langs
        cdef double reciprocal,non_zeroed

        ## check that model is not normalizeaad
        if model_ranks.normalized:
            self.logger.info('Resetting the model counts in training...')
            model_ranks.reset()
        
        ## update prediction count
        self.model_ranks.global_counts += 1.0

        ### global model
        scores = gen_scores.probs
        reranked_loc = gen_scores.first_rank
        non_zeroed = gen_scores.averaged_nonzeroed
        #reciprocal = ((1.0/(reranked_loc+1.0))*non_zeroed)
        reciprocal = 1.0/(reranked_loc+1.0)

        if reranked_loc < features.baseline or reranked_loc == 0.0: 
            global_ranks[0] += reciprocal
            global_langs[model_id] += reciprocal

        #print features.baseline

        for i in range(num_langs):

            ## the model of the current language
            if i == model_id:
                loc_scores = self.compute_instance(features,lang=i)
                scores = loc_scores.probs
                reranked_loc = loc_scores.first_rank
                non_zeroed = loc_scores.averaged_nonzeroed

            ## all other models 
            else:
                other_scores = self.compute_instance(features,lang=i)
                scores = other_scores.probs
                reranked_loc = other_scores.first_rank
                non_zeroed = other_scores.averaged_nonzeroed

            #reciprocal = 1.0/((reranked_loc+1.0)*non_zeroed)

            ## check if it better than the baseline model, only add if so
            if reranked_loc < features.baseline or reranked_loc == 0.0:
                                    
                #reciprocal = ((1.0/(reranked_loc+1.0))*non_zeroed)
                reciprocal = (1.0/(reranked_loc+1.0))
                ## add rank score for the overall dataset
                global_ranks[i+1] += reciprocal

                ## add rank score for the specific language under consideration 
                lang_ranks[model_id][i] += reciprocal

        try: 
            ## do update on the global model
            sgd_update(gen_scores,weights,lrate,0.0)

            ## do update on language specific model
            lang_counts[model_id] += 1.0
            lang_model = local_models[model_id]
            sgd_update(loc_scores,lang_model,lrate,0.0)

        except Exception,e:
            self.logger.error(e,exc_info=True)

        ## return the main model scores
        return <double>gen_scores.likelihood

    cdef void finish_iteration(self,int iteration,object wdir=None):
        """Normalize the model scores after complete iteration is completed

        :param iteration: the current iteration just finished
        :rtype: None 
        """
        cdef ModelProbs model_ranks = self.model_ranks
        ## normalize the model scores
        model_ranks.normalize(it=iteration,wdir=wdir,lang_map=self.lang_map)

    cdef void finish_evaluation(self):
        """Called after running on a testing set of some kind, passed by default
        
        :rtype: None 
        """
        cdef ModelProbs model_ranks = self.model_ranks
        self.logger.info('Finished evaluation, resetting the model scores...')
        model_ranks.reset()

    def __reduce__(self):
        ## pickle implementation
        return WeightedSGDPolyglot,(self._nfeatures,self.w,self.lang_models,self.hyper,self.lang_map)

cdef class AveragedWeightedSGDPolyglot(WeightedSGDPolyglot):
    """Averages the weights """

    cdef void finish_iteration(self,int iteration,object wdir=None):
        """Normalize the model scores after complete iteration is completed

        :param iteration: the current iteration just finished
        :rtype: None 
        """
        cdef ModelProbs model_ranks = self.model_ranks
        cdef double[:] weights = self.w
        cdef double[:,:] lmodels = self.lang_models
        cdef long num_feat = self._nfeatures

        ## normalize the model scores
        model_ranks.normalize(it=iteration,wdir=wdir,lang_map=self.lang_map)

        ## average the global vector
        poly_weighted_average(num_feat,weights,lmodels,model_ranks.global_ranks)
    
HYPER = {
    "simple_learn_rate": SimpleLearnRate
}

cpdef HyperParameters(config):
    """Returns a hyper parameter class (factory)

    :param config: the configuration to setup hyper parameters 
    :returns: a particular hyper parameter instance (loaded from config)
    """
    htype = config.hyper_type

    if not htype or htype not in HYPER: 
        raise ValueError('Unknown hyper parameter type: %s' % htype)
    return HYPER[htype].from_config(config)

## auxiliary methods

def find_lang_list(config):
    """Find the language list for the training data 

    :param config: the main configuration 
    :raises: ValueError
    """
    lang_file = config.atraining+".language"
    if not os.path.isfile(lang_file):
        raise ValueError('Cannot find the language file for model!: ' % lang_file)
    with open(lang_file) as lang_list:
        return {i:k for k,i in enumerate(set([z.strip() for z in lang_list.readlines()]))}

## print model probabilities

def print_model_scores(iteration,
                           wdir,
                           lang_scores,
                           polyglot_scores,
                           lang_id,
                           ptype,
                           lang_map,
                           sglobal=False,
                           ):
    """Print to file the model scores per language

    :param iteration: the current iteration 
    :param wdir: the current working direcotyr 
    :param lang_scores: the language scores 
    :param polyglot_scores: scores in the global model
    """
    prob_dir = os.path.join(wdir,"model_probs")
    if not os.path.isdir(prob_dir): os.mkdir(prob_dir)
    mfile = os.path.join(prob_dir,"probs_%d.txt" % iteration)
    rev_map = {value:key for key,value in lang_map.items()}

    ## print the scores 
    with open(mfile,ptype) as prob_file:

        ## language specific 
        if not sglobal: 
            print >>prob_file,"## %s" % str(rev_map.get(lang_id,lang_id))

            ## global scores
            print >>prob_file,"\tglobal=%s" % (polyglot_scores[lang_id])

            ## language specific
            for lang in range(lang_scores.shape[0]):
                print >>prob_file,"\t%s=%s" % (str(rev_map.get(lang,lang)),str(lang_scores[lang]))
                
        else:
            print >>prob_file,"## global"
            print >>prob_file,"\tglobal=%s" % (lang_scores[0])
            for lang in range(lang_scores.shape[0]-1):
                print >>prob_file,"\t%s=%s" % (str(rev_map.get(lang,lang)),str(lang_scores[lang+1]))
            

## c methods

## VANILLA PERCEPTRON UPDATE

@boundscheck(False)
@cdivision(True)
@wraparound(False)
cdef void poly_average(long num_features,double[:] gmodel,double[:,:] lmodels):
    """Averages the main global model overall this global model plus average models

    :param num_features: the number of total features 
    :param gmodel: the global model 
    :param lmodels: the local language models 
    """
    cdef long feature_id
    cdef double mvalue,lvalue,non_zero,wsum
    cdef double[:] ind_weights
    cdef int i,nmodels = lmodels.shape[0]

    for feature_id in range(num_features):
        mvalue = gmodel[feature_id]
        if mvalue == 0.0: continue
        ind_weights = lmodels[:,feature_id]
        non_zero = 1.0
        wsum = mvalue

        for i in range(nmodels):
            lvalue = ind_weights[i]
            if lvalue == 0.0: continue
            non_zero += 1.0
            wsum += lvalue

        ## take the average (note, the denominator is only on models
        ## where the feature weights have been observed)
        gmodel[feature_id] = wsum/non_zero


@boundscheck(False)
@cdivision(True)
@wraparound(False)
cdef void poly_weighted_average(long num_features,
                                   double[:] gmodel,
                                   double[:,:] lmodels,
                                   double[:] model_probs
                                   ):
    """Averages the global weight vector 


    :param num_feautres
    """
    cdef long feature_id
    cdef double mvalue,lvalue,non_zero,wsum
    cdef int i,nmodels = lmodels.shape[0]

    for feature_id in range(num_features):
        mvalue = gmodel[feature_id]
        if mvalue == 0.0: continue
        ind_weights = lmodels[:,feature_id]
        non_zero = 1.0
        wsum = mvalue

        for i in range(nmodels):
            lvalue = ind_weights[i]*model_probs[i+1]
            if lvalue == 0.0: continue
            non_zero += 1.0
            wsum += lvalue

        gmodel[feature_id] = wsum/non_zero
    

cdef void perceptron_update(FeatureAnalyzer current,int imposter,double[:] weights):
    """Vanilla Perceptron update rule

    :param current: the current example feature representation
    :param imposter: the index of the highest predicted value 
    :param weights: the current model weights 
    """
    cdef dict feature_scores = current.feature_scores
    cdef int[:] feature_ids  = np.array(feature_scores.keys(),dtype=np.int32)
    cdef double[:] fvlist
    cdef int feature_size = feature_ids.shape[0]
    cdef int i,j
    cdef long fid
    cdef double gold_v,imposter_v

    for i in range(feature_size):
        fid = feature_ids[i]
        fvlist = feature_scores[fid]
        gold_v = fvlist[0]
        imposter_v = fvlist[imposter]
        
        ## perceptron online update
        weights[fid] += <double>(gold_v - imposter_v)

cdef void aperceptron_update(FeatureAnalyzer current,int imposter,
                                 double[:] weights,double[:] counts, ## copies of feature vector
                                 int[:] last_update, # list of last updates
                                 int num_updates ## total number of updates so far
                                 ):
    """Average perceptron update rule

    :param current: the current example feature representation 
    :param imposter: the id of the highest ranked item (assumes it is not the gold!) 

    :param weights: the model parameters/weights 
    :param counts: the cummulative counts of the features 
    :param last_update: the last time each feature was updated 
    """
    cdef dict feature_scores = current.feature_scores
    cdef int[:] feature_ids = np.array(feature_scores.keys(),dtype=np.int32)
    cdef double[:] fvlist
    cdef int feature_size = feature_ids.shape[0]
    cdef int i,j
    cdef long fid
    cdef double gold_v,imposter_v
    cdef double new_value
    cdef int gap

    for i in range(feature_size):
        fid = feature_ids[i]
        fvlist = feature_scores[fid]
        gold_v = fvlist[0]
        imposter_v = fvlist[imposter]
        new_value = gold_v - imposter_v

        ## gap since last update (update count as needed)
        gap = num_updates - last_update[fid]
        if gap > 1: counts[fid] += weights[fid]*float(gap-1)

        # ## standard update 
        weights[fid] += new_value

        # ## update counts 
        counts[fid] += new_value

        ## update last update
        last_update[fid] = num_updates

## VANILLA SGD UPDATE 

#cdef void sgd_update(FeatureAnalyzer current,double[:] weights,double lrate,double rlambda):
cdef int sgd_update(FeatureAnalyzer current,double[:] weights,double lrate,double rlambda) except -1:
    """(Vanilla) Stochastic gradient update function
    
    :param current: the current example information, probabilities, etc..

    :param weights: the model parameters or weights

    :param lrate: the learning rate 
    :param rlambda: the regularizer term
    """
    cdef double[:] probs     = current.probs
    cdef dict feature_scores = current.feature_scores
    cdef int[:] feature_ids  = np.array(feature_scores.keys(),dtype=np.int32)
    cdef double[:] fvlist 
    cdef int feature_size    = feature_ids.shape[0]
    cdef int i,j
    cdef long fid
    cdef double exp_counts,emp_counts
    cdef int size = current.size+1

    ## go through each feature ()
    for i in range(feature_size):
        fid = feature_ids[i]
        fvlist = feature_scores[fid]
        
        exp_counts = 0.0
        emp_counts = 0.0

        ## expected counts
        for j in range(size):
            exp_counts += probs[j]*fvlist[j]
            
        ## expected counts (gold value, also at index 0)
        emp_counts = fvlist[0]
                
        ## online update
        ## note : regularization is not being done correctly here 
        weights[fid] += <double>(lrate*(emp_counts - exp_counts - (rlambda*weights[fid])))



### regularized update

## not implemented correctly 
        
cdef void rsgd_update(FeatureAnalyzer current, ## feature representation
                          double[:] weights,int[:] reg_time,int updates, ## weights and time stamps
                          double[:] lrates,
                          double lrate,double rlambda ## hyper parameters
                          ):
    """(Vanilla) Stochastic gradient update function
    
    :param current: the current example information, probabilities, etc..

    :param weights: the model parameters or weights

    :param lrate: the learning rate 
    :param rlambda: the regularizer term
    """
    cdef double[:] probs     = current.probs
    cdef dict feature_scores = current.feature_scores
    cdef int[:] feature_ids  = np.array(feature_scores.keys(),dtype=np.int32)
    cdef double[:] fvlist 
    cdef int feature_size    = feature_ids.shape[0]
    cdef int i,j
    cdef long fid
    cdef double exp_counts,emp_counts
    cdef int size = current.size+1
    cdef int gap

    ## go through each feature ()
    for i in range(feature_size):
        fid = feature_ids[i]
        fvlist = feature_scores[fid]
        exp_counts = 0.0
        emp_counts = 0.0

        ## expected counts
        for j in range(size):
            exp_counts += probs[j]*fvlist[j]
                        
        ## expected counts (gold value, also at index 0)
        emp_counts = fvlist[0]

        ## update regularization (for that haven't been updated in a while)
        gap = reg_time[fid]
        for j in range(gap,updates-1):
            weights[fid] += lrates[j]*(0.0 - (rlambda*weights[fid]))

        ## online update
        weights[fid] += <double>(lrate*(emp_counts - exp_counts - (rlambda*weights[fid])))

        ## update the regularization time 
        reg_time[fid] = updates

## slow/naive regularized update

cdef void reg_sgd_update(FeatureAnalyzer current,double[:] weights,
                             long num_features, # the number of features in mode
                             double lrate,double rlambda ## hyper parameters
                             ):
    """(Vanilla) Stochastic gradient update function
    
    :param current: the current example information, probabilities, etc..

    :param weights: the model parameters or weights

    :param lrate: the learning rate 
    :param rlambda: the regularizer term
    """
    cdef double[:] probs     = current.probs
    cdef dict feature_scores = current.feature_scores
    cdef int[:] feature_ids  = np.array(feature_scores.keys(),dtype=np.int32)
    cdef double[:] fvlist 
    cdef int feature_size    = feature_ids.shape[0]
    cdef long fid
    cdef double exp_counts,emp_counts
    cdef int size = current.size+1
    cdef long i,j

    for i in range(num_features):
        
        if i not in feature_scores:
            exp_counts = 0.0
            emp_counts = 0.0
        else:
            fvlist = feature_scores[i]
            exp_counts = 0.0
            emp_counts = 0.0

            for j in range(size):
                exp_counts += probs[j]*fvlist[j]

            emp_counts = fvlist[0]

        weights[i] += <double>(lrate*(emp_counts - exp_counts - (rlambda*weights[i])))

        
## AVERAGED SGD UPDATE 

cdef void asgd_update(FeatureAnalyzer current,
                          double[:] weights,double[:] counts, ## parameters and parameter counts
                          int[:] last_update,int num_updates, # record of last updates
                          int iteration, ## the current iteration
                          double lrate,double rlambda ## hyper parameters
                          ):
    """(Vanilla) Stochastic gradient update function
    
    :param current: the current example information, probabilities, etc..

    :param weights: the model parameters or weights

    :param lrate: the learning rate 
    :param rlambda: the regularizer term
    """
    cdef double[:] probs     = current.probs
    cdef dict feature_scores = current.feature_scores
    cdef int[:] feature_ids  = np.array(feature_scores.keys(),dtype=np.int32)
    cdef double[:] fvlist 
    cdef int feature_size    = feature_ids.shape[0]
    cdef int i,j
    cdef long fid
    cdef double exp_counts,emp_counts,averaged_weight
    cdef int size = current.size+1
    cdef double update_value 

    ## go through each feature
    for i in range(feature_size):
        fid = feature_ids[i]
        fvlist = feature_scores[fid]
        exp_counts = 0.0
        emp_counts = 0.0

        ## expected counts
        for j in range(size):
            exp_counts += probs[j]*fvlist[j]
            
        ## expected counts (gold value, also at index 0)
        emp_counts = fvlist[0]

        gap = num_updates - last_update[fid]
        if gap > 1: counts[fid] += weights[fid]*float(gap-1)

        ## online update
        ## note: regularization is not being done correctly here 
        update_value = lrate*(emp_counts - exp_counts - (rlambda*weights[fid]))
        weights[fid] += update_value
        
        ## online update
        counts[fid] += update_value
        last_update[fid] = num_updates


cdef FeatureAnalyzer weighted_train_prob(FeatureObj features,double[:] weights,bint norm=True):
    """Computes the train probability, while also keep track of average number of 
    non_zero valued features used to compute score"""
    cdef int i,j

    ## gold features
    cdef Vectorizer gold_features = features.vectorize_gold()
    cdef long[:]    gold_ids  = gold_features.features
    cdef double[:]  gold_vals = gold_features.feat_counts
    cdef int        gold_size = gold_features.size

    ## other features
    cdef Vectorizer other_features
    cdef long[:]    fids
    cdef double[:]  fvals
    cdef int        size

    ## feature analyzer
    cdef int beam = features.beam
    cdef FeatureAnalyzer analysis = FeatureAnalyzer(beam)

    ## nonzeroed
    cdef double[:] non_zeroed = np.zeros((beam+1,),dtype='d')

    ##  scores
    cdef double feature_score,feature_count
    cdef long   feature_name

    for i in range(gold_size):
        feature_name  = gold_ids[i]
        feature_count = gold_vals[i]
        feature_score = feature_count*weights[feature_name]
        analysis.add_gold_feature(feature_name,feature_count,feature_score)
        if weights[feature_name] != 0.0: non_zeroed[0] += 1.0

    ## other features 
    for i in range(beam):
        other_features = features.vectorize_item(i)
        fids  = other_features.features
        fvals = other_features.feat_counts
        size = other_features.size

        # ## individual features 
        for j in range(size):
            feature_name  = fids[j]
            feature_count = fvals[j]
            feature_score = feature_count*weights[feature_name]
            analysis.add_feature(i,feature_name,feature_count,feature_score)
            if weights[feature_name] != 0.0: non_zeroed[i+1] += 1.0

    ## normalize and compute likelihood (if needed)
    analysis.normalize()

    ## find average of non_zeroed
    #analysis.averaged_nonzeroed = max(0.01,log(np.sum(non_zeroed)/float(beam+1.0)))
    analysis.averaged_nonzeroed = max(0.01,np.sum(non_zeroed)/float(beam+1.0))

    ## return the feature analysis 
    return analysis


        
        
cdef FeatureAnalyzer train_prob(FeatureObj features,double[:] weights,bint norm=True):
    """Compute the score for a given example (k-best list), and computes features counts for 
    gold and rest.

    :param features: the example feature representation 
    :param weights: the model parameters
    """

    cdef int i,j

    ## gold features
    cdef Vectorizer gold_features = features.vectorize_gold()
    cdef long[:]    gold_ids  = gold_features.features
    cdef double[:]  gold_vals = gold_features.feat_counts
    cdef int        gold_size = gold_features.size

    ## other features
    cdef Vectorizer other_features
    cdef long[:]    fids
    cdef double[:]  fvals
    cdef int        size

    ## feature analyzer
    cdef int beam = features.beam
    cdef FeatureAnalyzer analysis = FeatureAnalyzer(beam)

    ##  scores
    cdef double feature_score,feature_count
    cdef long   feature_name

    ## gold features (if there)
    
    for i in range(gold_size):
        feature_name  = gold_ids[i]
        feature_count = gold_vals[i]
        feature_score = feature_count*weights[feature_name]
        analysis.add_gold_feature(feature_name,feature_count,feature_score)

    ## other features 
    for i in range(beam):
        other_features = features.vectorize_item(i)
        fids  = other_features.features
        fvals = other_features.feat_counts
        size = other_features.size

        # ## individual features 
        for j in range(size):
            feature_name  = fids[j]
            feature_count = fvals[j]
            feature_score = feature_count*weights[feature_name]
            analysis.add_feature(i,feature_name,feature_count,feature_score)

    ## normalize and compute likelihood (if needed)
    if norm: analysis.normalize()
    return analysis


### standard evaluator
#######################

cdef double weighted_scoring(int point,FeatureObj features,
                         RankComparison new_ranks,
                         double[:] weights,
                         int lang_id,
                         double[:,:] local_models,
                         double[:] lang_distr) except -1:
    """Reranks a k-best list by voting using a number of different models weighted 
    by training 

    :param features: the feature for the baseline k-best list 
    :param new_ranks: the new rank items to fill up from reranking and voting
    :param weights: the global/polyglot model parameters
    :param model_id: the identity of the target language
    :param local_models: the local models 
    :param lang_distr: the distribution over models given the select language type language 
    """
    cdef int i,j

    cdef Vectorizer other_features
    cdef long[:]    fids
    cdef double[:]  fvals
    cdef int        size

    cdef int beam = features.beam

    ### global reranked representation
    cdef FeatureAnalyzer analysis = FeatureAnalyzer(beam-1)
    cdef double[:] probs = analysis.probs
    
    ## local reranked representation

    cdef int model,num_models = local_models.shape[0]
    cdef double model_prob
    cdef double[:] model_parameters
    
    ## scores
    cdef double feature_score,feature_count
    cdef long   feature_name

    ## storage and gold positions
    cdef RankStorage storage = new_ranks.storage
    cdef int[:] gold_pos = storage.gold_pos
    cdef int gold_loc = gold_pos[point]
    
    ## ranks (need to do this directly)
    cdef int[:,:] nranks = new_ranks.new_ranks
    cdef int[:] new_rank_list = nranks[point]
    cdef int[:] old_ranks = new_ranks.old_ranks(point)

    ## for some reason, the argsort convert to 
    cdef long[:]  new_ordering
    cdef long num_params = weights.shape[0]
    cdef int total_size = old_ranks.shape[0]

    ## first with global model
    for model in range(num_models):
        model_prob = lang_distr[model]
        
        ## global polyglot model
        if model == 0: model_parameters = weights
        ## non global, local model
        else: model_parameters = local_models[model-1]

        ## holder for reranked list 
        local_analysis = FeatureAnalyzer(beam-1)
        ## probabilities of items in list 
        local_probs = local_analysis.probs

        for i in range(beam):
            other_features = features.vectorize_item(i)
            fids  = other_features.features
            fvals = other_features.feat_counts
            size  = other_features.size

            for j in range(size):
                feature_name  = fids[j]
                feature_count = fvals[j]
                feature_score = feature_count*model_parameters[feature_name]
                local_probs[i] += feature_score

        ## normalize to get probabilities over all items in list
        local_analysis.normalize()

        ## add each probability to global scores
        for i in range(beam):
            probs[i] += (local_probs[i]*model_prob)

    ## normalize the final list in order to see resulting likelihood
    try: 
        analysis.normalize(start=gold_loc)
    except:
        pass

    ## sort  the global probability list
    new_ordering = np.argsort(probs)[::-1]

    # ## put the result in normal order again
    for i in range(total_size):
        if i >= beam: new_rank_list[i] = old_ranks[i]
        else: new_rank_list[i] = old_ranks[new_ordering[i]]

    return analysis.likelihood
        

cdef FeatureAnalyzer score(int point,FeatureObj features,RankComparison new_ranks, ## feature and current ranks 
                                   double[:] weights,bint norm=True):
    """Computes probabilities using model for evaluation purposes

    :param point: the point in the data 

    :param features: the example feature representation 
    :param new_ranks: the rank comparison with pointers to baseline ran

    :param weights: the model parameters

    """
    cdef int i,j

    cdef Vectorizer other_features
    cdef long[:]    fids
    cdef double[:]  fvals
    cdef int        size
    
    cdef int beam = features.beam
    #cdef FeatureAnalyzer analysis = FeatureAnalyzer(beam)
    cdef FeatureAnalyzer analysis = FeatureAnalyzer(beam-1)
    cdef double[:] probs = analysis.probs

    ## scores
    cdef double feature_score,feature_count
    cdef long   feature_name

    ## storage and gold positions
    cdef RankStorage storage = new_ranks.storage
    cdef int[:] gold_pos = storage.gold_pos
    cdef int gold_loc = gold_pos[point]
    
    ## ranks (need to do this directly)
    cdef int[:,:] nranks = new_ranks.new_ranks
    cdef int[:] new_rank_list = nranks[point]
    cdef int[:] old_ranks = new_ranks.old_ranks(point)

    ## for some reason, the argsort convert to 
    cdef long[:]  new_ordering
    cdef long num_params = weights.shape[0]

    cdef int total_size = old_ranks.shape[0]
    
    ## go through candidates
    for i in range(beam):
        other_features = features.vectorize_item(i)
        fids  = other_features.features
        fvals = other_features.feat_counts
        size  = other_features.size

        ## individual featrures
        for j in range(size):
            feature_name  = fids[j]
            feature_count = fvals[j]
            ### possible error here 
            #if feature_name > num_params: continue
            feature_score = feature_count*weights[feature_name]
            probs[i] += feature_score

        ## divide by number of features?
        #probs[i] = probs[i]/float(size)

    ## normalize probs, compute likelihood and rerank
    if norm:
        ## if gold is too far down rank list, this will raise
        ## an exception, and assign 0.0 likelihood (as it should)
        try: 
            analysis.normalize(start=gold_loc)
        except:
            pass

    new_ordering = np.argsort(probs)[::-1]

    ## aligned the reranked list with original values
    
    # for i in range(beam):
    #     new_rank_list[i] = old_ranks[new_ordering[i]]

    for i in range(total_size):
        if i >= beam: new_rank_list[i] = old_ranks[i]
        else: new_rank_list[i] = old_ranks[new_ordering[i]]

    ## assign to
    return analysis



#### AVERAGE EVALUATOR

cdef FeatureAnalyzer ascore(int point,FeatureObj features,RankComparison new_ranks, ## feature and current ranks 
                                   double[:] counts,int num_updates,bint norm=True):
    """Computes scores for examples using average weights

    :param point: the point in the data 
    :param features: the example feature representation 
    :param new_ranks: the rank comparison with pointers to baseline ran

    :param weights: the model parameters

    """
    cdef int i,j

    cdef Vectorizer other_features
    cdef long[:]    fids
    cdef double[:]  fvals
    cdef int        size
    
    cdef int beam = features.beam
    #cdef FeatureAnalyzer analysis = FeatureAnalyzer(beam)
    cdef FeatureAnalyzer analysis = FeatureAnalyzer(beam-1)
    cdef double[:] probs = analysis.probs

    ## scores
    cdef double feature_score,feature_count
    cdef long   feature_name

    ## storage and gold positions
    cdef RankStorage storage = new_ranks.storage
    cdef int[:] gold_pos = storage.gold_pos
    cdef int gold_loc = gold_pos[point]
    
    ## ranks (need to do this directly)
    cdef int[:,:] nranks = new_ranks.new_ranks
    cdef int[:] new_rank_list = nranks[point]
    cdef int[:] old_ranks = new_ranks.old_ranks(point)

    ## for some reason, the argsort convert to 
    cdef long[:]  new_ordering

    cdef int total_size = old_ranks.shape[0]
    
    ## go through candidates
    #for i in beam(range):
    for i in range(beam):
        other_features = features.vectorize_item(i)
        fids  = other_features.features
        fvals = other_features.feat_counts
        size  = other_features.size

        ## individual featrures
        for j in range(size):
            feature_name  = fids[j]
            feature_count = fvals[j]
            feature_score = feature_count*(counts[feature_name]/float(num_updates))
            probs[i] += feature_score

        ## divide by number of features?
        #probs[i] = probs[i]/float(size)

    ## normalize probs, compute likelihood and rerank
    if norm: analysis.normalize(start=gold_loc)
    new_ordering = np.argsort(probs)[::-1]

    ## aligned the reranked list with original values
            
    # for i in range(beam):
    #     new_rank_list[i] = old_ranks[new_ordering[i]]

    for i in range(total_size):
        if i >= beam: new_rank_list[i] = old_ranks[i]
        else: new_rank_list[i] = old_ranks[new_ordering[i]]

    ## assign to
    return analysis
