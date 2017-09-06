# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementation of Optimizers 

"""
import sys
import time
import os
import logging
import traceback
import pickle
import numpy as np
cimport numpy as np
from zubr.util.config import ConfigAttrs
from zubr.util.optimizer_util import build_optimizer,find_extractor,finish_op
from zubr.util.optimizer_util import find_data,restore_config,backup_current,reload_model
from zubr.Dataset   cimport RankDataset,RankPair,RankComparison,RankScorer
from zubr.Features  cimport FeatureObj
from zubr.Extractor cimport Extractor
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Learner   cimport Learner,OnlineLearner
from copy import deepcopy

OPTIMIZERS = {
    ## old 
    "online"     : OnlineOptimizer,
    #"new_online" : OnlinePipelineOptimizer,
    #"batch"     : BatchOptimizer,
    #minibatch   : MiniBatchOptimizer,
}

cdef class OptimizerBase(ZubrSerializable):
    """Base class for optimization implementations"""
    pass
 
cdef class RankOptimizer(OptimizerBase):

    """Optimizers that involve ranking or classification

    -- These classes all use a zubr.util.config.ConfigAttr configuration 
    instances to specify all of the optimizers settings, and the location
    of the target data. 

    -- Each configuration must have a ``config.dir`` attribute, which specifies
    the location of the working directory. This is where all the data for training,
    evaluating, and tuning the optimizer should sit. Standardly, the datasets should 
    be named train.data,valid.data (optional), and test.data (optional) for train, 
    validation/tuning, and testing data respestively. 

    -- Building from configuration can be done using: cls.from_config(config)

    -- Each optimizer instance has an ``learner`` attribute, which specifies the
    type  of learning model being used (e.g., sgd, perceptron, neural network, ),
    the type of update rules, etc... In other words, the learner defines the actual
    optimization procedure and model parameters. 


    -- assumes the following datasets: train,valid,test -- will monitor results on
    the validation set when training and backup the best performing models. 

    """
    
    cdef int _train_model(self,RankDataset dataset,RankDataset valid) except -1:
        """The main method for training the linear model

        :param dataset: the target dataset 
        :param valid: the validation set (optional)
        """
        raise NotImplementedError

    cdef RankScorer _test_model(self,RankDataset dataset,str ttype,int it=-1,bint debug=True):
        """The main method for training the linear model

        :param dataset: the target dataset 
        :param ttype: the type of data testing (e.g., train/validation,..)
        :param it: the number of iterations run (optional, -1 default)
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls,config):
        """Build an optimizer from a configuration. 

        :param config: optimizer or experiment configuration
        :returns: SimpleOptimizer instance 
        """
        raise NotImplementedError

    def train(self,train_type='train'):
        """Optimize a model using training data

        :rtype: None 
        """
        self.logger.info('starting training with: %s' % train_type)
        dataset,valid = find_data(self.config,train_type)
        self._train_model(dataset,valid)

    def test(self,test_type='test'):
        """Evaluate a model on testing data 

        :param test_type: the type of dataset to evaluate on
        :rtype: None 
        """
        iters = (self.config.epochs-1)
        self.logger.info('starting evaluating with: %s' % test_type)
        dataset = find_data(self.config,test_type)
        self._test_model(dataset,test_type,iters)
        
    ## Properties 
    
    property epochs:
        """The number of epochs or iterations used to train mode"""

        def __get__(self):
            """Returns the number of epochs 

            :rtype: int
            """
            return <int>self.config.epochs

        def __set__(self,int new_epochs):
            """Sets the number of epochs 

            :param new_epochs: int
            :rtype: None 
            """
            self.config.epochs = new_epochs

    ## keep track of improvements per iteration

    property improvement:
        """Keeps track of improvements made on validation (if validation available)"""

        def __get__(self):
            """Returns the number of iteration from last improvement

            :rtype: int 
            """
            return self._improvement 

        def __set__(self,int new_val):
            """Updates last improvement 

            :param new_val: the new iteration where improvement is
            :rtype: None 
            """
            self._improvement = new_val

    property best_model:
        """Keep a copy of the best performing model"""

        def __get__(self):
            """Returns the number of iteration from last improvement

            :rtype: int 
            """
            return self._best_model 

        def __set__(self,new_best):
            """Updates last improvement 

            :param new_val: the new iteration where improvement is
            :rtype: None 
            """
            self._best_model = new_best

    ## exit code

    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        ## passes by default 
        self.exit()

    def exit(self):
        if hasattr(self.extractor,'exit'):
            self.logger.info('Exiting the extractor')
            self.extractor.exit()

cdef class OnlineOptimizer(RankOptimizer):

    """A optimizer that works ``online``, where updates to the model during 
    learning are performed at each example in the training data.
    """

    def __init__(self,config,extractor,model,vscore):
        """Initialize the online optimizer  
        
        :param config: the optimizer configuration 
        :param extractor: the feature extractor 
        :param model: the feature vector 
        :param vscore: the score on validation set (if monitored)
        """
        self.config       = config
        self.extractor    = extractor
        self.model        = model
        self.vscore       = vscore
        self._improvement = 0
        self._best_model  = False

    property shuffle:
        """Shuffle parameter for dataset when training"""

        def __get__(self):
            """Returns is shuffling is used or not

            :rtype: bool
            """
            return <bint>self.config.shuffling

        def __set__(self,bint new_value):
            """Sets the shuffling parameter

            :param new_value: the new shuffling parameter
            :rtype: None 
            """
            self.config.shuffling = new_value

    @classmethod
    def from_config(cls,config):
        """Build an online optimizer from a zubr configuration


        :param config: the optimizer and experiment configuration 
        :type config: zubr.util.config.ConfigAttrs
        :returns: an online optimizer
        :rtype: OnlineOptimizer
        """
        settings_and_options = ConfigAttrs()
        extractor = find_extractor(config)
        build_optimizer(config,settings_and_options)
        config.num_features = extractor.num_features
        learner = Learner(config)
        vscore  = RankScorer(0.0,0.0,0.0)
        return cls(settings_and_options,extractor,learner,vscore)

    def __reduce__(self):
        ## pickle implementation
        return OnlineOptimizer,(self.config,self.extractor,self.model,self.vscore)

    cdef inline void log_iter(self,i,start_time,ll):
        """Log information about iteration.

        :param i: iteration number 
        :param start_time: the starting time 
        :param ll: the likelihood
        :rtype: None 
        """
        self.logger.info(
            'Finished iteration %d in %f seconds, log-likelihood=%f' %\
            (i,time.time()-start_time,ll))

    cdef inline void log_test(self,i,ttype,start_time,ll):
        """Log information about testing run.

        :param i: the current number of iterations
        :param ttype: the type of testing
        :param start_time: the starting time
        :param ll: the log likelihood
        :rtype: None
        """
        self.logger.info(
            'Finished testing model on <%s> in %f seconds after %d iters, log-likelihood=%f' %\
            (ttype,time.time()-start_time,i+1,ll))

    cdef RankScorer _test_model(self,RankDataset dataset,str ttype,int it=-1,bint debug=True):
        """Test your model on a given dataset.

        :param dataset: the dataset to evaluate on 
        :param ttype: the type of data (e.g., train, validation, test, ...) 
        :returns: Nothing        
        """
        cdef Extractor extractor = <Extractor>self.extractor
        cdef object config = self.config
        cdef int size = dataset.size,data_point

        ## likelihood
        cdef double likelihood   = 1e-1000
        
        ## rank information 
        cdef RankPair instance
        cdef FeatureObj features
        cdef RankComparison new_ranks

        ## the underlying model 
        cdef OnlineLearner model = <OnlineLearner>self.model
        ## working directory
        cdef str wdir = config.dir
        cdef double score
        
        ## offline extract (builds ranks)
        extractor.offline_init(dataset,ttype)

        ## new ranks to generate 
        new_ranks = extractor.rank_init(ttype)

        ## test start time
        if debug: self.logger.info('Starting testing  loop...')
        t_start_time = time.time()

        for data_point in range(size):
            
            instance = dataset.get_item(data_point)

            ## there's an issue here 
            features = extractor.extract(instance,ttype)

            ## score with model
            score = model.score_example(instance,data_point,features,new_ranks)
            likelihood += score

        ## log time 
        if debug: self.log_test(it,ttype,t_start_time,likelihood)

        ## call
        model.finish_evaluation()

        ## processing after iteration
        if dataset.multilingual:
            return new_ranks.multi_evaluate(dataset,ttype,wdir,it=it,ll=likelihood,debug=debug)
        return <RankScorer>new_ranks.evaluate(ttype,wdir,it=it,ll=likelihood,debug=debug)
    
    cdef int _train_model(self,RankDataset dataset,RankDataset valid) except -1:
        """Training a model using a Rank dataset and stochastic gradient descent.

        :param dataset: the target dataset 
        :param valid: the validation set (optional)
        """
        cdef Extractor extractor = <Extractor>self.extractor
        cdef object config = self.config
        cdef int epochs = config.epochs
        cdef int iteration,data_point
        cdef int size = dataset.size,gold,new_id
        cdef double likelihood
        cdef bint shuffle = config.shuffle

        ## data instance from training 
        cdef RankPair instance

        ## feature representation 
        cdef FeatureObj features
        ## underlying model 
        cdef OnlineLearner model = <OnlineLearner>self.model

        ## evaluate on train in the end?
        cdef bint eval_train = config.eval_train

        ## current scores
        cdef RankScorer r,vscore

        ## the working directory
        cdef object wdir = config.dir

        ## initialize extractor (if needed)
        extractor.offline_init(dataset,'train')

        for iteration in range(epochs):

            likelihood   = 1e-1000
            i_start_time = time.time()
            if shuffle: dataset.shuffle()
            self.logger.info('Starting training loop...')

            for data_point in range(size):

                ## log current position
                if (data_point+1 % 5000) == 0: self.logger.info('Currently at point %d' % data_point)

                ## extracts feature representation for training instance
                instance = dataset.next_ex()
                features = extractor.extract(instance,'train')

                ## performs update
                score = model.online_update(instance,features,data_point,iteration,size)
                
                ## update likelihood (if important) 
                likelihood += score

            model.finish_iteration(iteration,wdir)
            self.log_iter(iteration+1,i_start_time,likelihood)
           
            ## test current model on validation (if available)
            if not valid.is_empty:

                ## validation score 
                r = self._test_model(valid,'valid',iteration)

                ## check if score improved 
                vscore = self.vscore
                if r > vscore:

                    ## increment improvement iteration 
                    self.improvement = iteration
                    ## score model
                    self.vscore = r
                    ## back up the current model as best model 
                    self.best_model = deepcopy(self.model)
                    
                ## stop training if improvement hasn't improved in a while 
                elif (iteration - self.improvement) > 3:
                    self.logger.info('Stopping early after %s iterations' % str(iteration+1))
                    break

        ## make sure that current model is the best model
        if self.improvement != iteration and self.best_model:
            self.logger.info('Swapping with best model: %s' % str(self.improvement+1))
            self.model = self.best_model
            self.best_model = False

    ## backup protocol

    def backup(self,wdir):
        """Backup the optimizer model

        :param wdir: the working directory
        """
        stime = time.time()

        ## the learning models
        lpath = os.path.join(wdir,"optimizer_model")
        if os.path.isdir(lpath):
            self.logger.info('Already backed up, skipping...')
            return

        ## make the directory 
        os.mkdir(lpath)
        
        ## back up the model
        self.logger.info('Backing up the model components....')
        self.model.backup(lpath)
        
        ## the extractor model
        self.logger.info('Backing up extractor (if needed..)')
        self.extractor.backup(wdir)

        ## vscore item
        vpath = os.path.join(lpath,"vscores")
        np.savez_compressed(vpath,np.array([self.vscore.at1,self.vscore.at10,self.vscore.mrr]))

        ## dump the configuration 
        self.logger.info('Backing up the config...')
        econfig = os.path.join(lpath,"optimizer_config.p")
        with open(econfig,'w') as my_config:
            pickle.dump(self.config,my_config)

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load an optimizer instance from backup file

        :param config: the main configuration 
        """
        stime = time.time()
        uwdir = config.dir 
        lpath = os.path.join(config.dir,"optimizer_model")

        ## load the extractor
        extractor = find_extractor(config)

        ## load the model
        config.dir = lpath
        lclass = Learner(config)
        model = lclass.load_backup(config)
        config.dir = uwdir 

        ## generic score
        vpath = os.path.join(lpath,"vscores.npz")
        points = np.load(vpath)["arr_0"]
        vscore  = RankScorer(points[0],points[1],points[2])

        ## the configruation
        with open(os.path.join(lpath,"optimizer_config.p"),'rb') as config:
            settings = pickle.load(config)

        instance = cls(settings,extractor,model,vscore)
        instance.logger.info('Loaded in %s seconds' % str(time.time()-stime))
        return instance


cdef class MiniBatchOptimizer(RankOptimizer):
    pass 

cdef class BatchOptimizer(RankOptimizer):

    """A optimizer that works using ``batch`` updates, i.e., each change to 
    the model during training is performed after running through the full data. 
    """
    pass

def Optimizer(otype):
    """The type of optimizer to use

    :param otype: type of optimizer
    :type optype: str
    """
    olower = otype.lower()
    if olower not in OPTIMIZERS:
        raise ValueError('Unknown optimizer type: %s' % olower)
    return OPTIMIZERS[olower]

### CLI SETTINGS

def params():
    """main parameters for training a discriminative model

    :rtype: tuple
    :returns: descriptions of options with names, list of options
    """
    options = [
        ("--miters","miters",5,"int",
         "number of iterations/epochs [default=5]","Optimizer"),
        ("--optim","optim","online","str",
         "type of optimizer to use [default=sgd]","Optimizer"),
        ("--learner","learner","lsgd","str",
         "type of learning model to use [default=lsgd]","Optimizer"),         
        ("--shuffle","shuffle",True,"bool",
        "shuffle the order (for online learning) [default=True]","Optimizer"),
        ("--rlambda","rlambda",0.0,"float",
        "the regularization lambda term [default=0.1]","HyperParameters"),
        ("--lrate1","lrate1",0.1,"float",
         "learning rate parameter 1  [default=0.1]","HyperParameters"),
        ("--lrate2","lrate2",0.001,"float",
         "learning rate parameter 2  [default=0.001]","HyperParameters"),
        ("--eval_val","eval_val",False,"bool",
         "Evaluate on validation when training [default=False]","Optimizer"),
        ("--eval_train","eval_train",False,"bool",
         "Evaluate on training data after training [default=False]","Optimizer"),
        ("--eval_test","eval_test",False,"bool",
         "Evaluate on testing data [default=False]","Optimizer"),
        ("--hyper_type","hyper_type","simple_learn_rate","str",
         "The type of hyper parameters [default='simple_learn_rate']","Optimizer"),
        ("--model_loc","model_loc","","str",
         "The location of trained model [default='simple_learn_rate']","Optimizer"),
        ("--restore","restore","","str",
         "Location of optimizer data [default='']","Optimizer"),
        ("--hidden_layers","hidden_layers",3,"int",
         "Number of hidden layers (neural) [default=3]","NeuralNetwork"),
        ("--size_output","size_output",2,"int",
         "The size of output layer [default=2]","NeuralNetwork"),
        ("--non_linear","non_linear",'tanh',"str",
         "The non-linear function to apply [default='tanh']","NeuralNetwork"),
        ("--hidden_neurons","hidden_neurons",20,"int",
         "The number of hidden nodes [default=20]","NeuralNetwork"),
        ("--retrain_indv","retrain_indv",False,"bool",
         "retrain model after removing individiual features [default=20]","Optimizer"),
        ("--retrain_temp","retrain_temp",False,"bool",
         "retrain model after removing templates [default=20]","Optimizer"),         
     ]

    model_group = {"Optimizer":"settings for optimization"}
    model_group["NeuralLearner"] = "Neural network settings"
    model_group["LinearLearner"] = "Settings for linear models"
    model_group["HyperParameters"] = "Hyper parameters settings"
    return (model_group,options)

def argparser():
    """Returns an aligner argument parser using default

    :rtype: zubr.util.config.ConfigObj 
    :returns: default argument parser 
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr optimizer [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser


def main(argv):
    """Main point of execution

    :param config: the main configuration for running optimizer
    """
    
    try:
        ## configuration
        from_scratch = False
        
        if isinstance(argv,ConfigAttrs):
            config = argv

        else:
            parser = argparser()
            config = parser.parse_args(argv[1:])
            logging.basicConfig(level=logging.DEBUG)

        ## previous model specified to restore,
        if config.restore:

            dloc = None
            ## override config model_loc if specific
            if not config.retrain_temp and not config.retrain_indv:
                if config.model_loc: dloc = config.model_loc
                restore_config(config,config.restore)
                if dloc: config.model_loc = dloc

            optimizer_class = Optimizer(config.optim)

            ## reload the model 
            if not config.pipeline_backup: 
                optimizer = reload_model(config,optimizer_class.load_large)
            else:
                optimizer = optimizer_class.load_backup(config)
            
            ### RETRAIN MODEL
            
            if config.retrain_indv or config.retrain_temp:
                optimizer.logger.info('Retraining the model with model %s....' % config.model_loc)
                optimizer.train()

            ## EVALUATE ON A DATASET
            elif config.eval_val:
                optimizer.test(test_type='valid')
            elif config.eval_train:
                optimizer.test(test_type='train-test')
            elif config.eval_test:
                optimizer.test(test_type='test')

        ## otherwise trains a model from scratch 
        else:
            
            from_scratch = True

            ## optimizer class
            optimizer_class = Optimizer(config.optim)
            #optimizer = optimizer_class.from_config(config)
            with optimizer_class.from_config(config) as optimizer: 
                optimizer.train()

                ## test the resulting model 
                if config.eval_test:
                    optimizer.test(test_type='test')

    except Exception,e:
        traceback.print_exc(file=sys.stdout)
    
    finally:
        try:
            if not config.pipeline_backup:
                finish_op(config,optimizer.config,optimizer.dump_large,from_scratch)
            else:
                ## back up the model 
                optimizer.backup(config.dir)
                ## create eval script
                finish_op(config,optimizer.config,None,from_scratch)
        except Exception,e:
            traceback.print_exc(file=sys.stdout)

        ## shut off optimizer 
