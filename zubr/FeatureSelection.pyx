# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Performs feature selection for LinearRankOptimizer models

By default, it will look at results from validation set, 
and greedily remove redundant and high-valued features that 
occur when the model makes mistakes. If these features improve
the validation score, then the they are removed from the model. 

"""

import logging
import sys
import time
import copy
import traceback
from zubr.util.config import ConfigAttrs
from zubr.util.selector_util import restore_config,backup_better,wrong_features
from zubr.Dataset cimport RankScorer,RankComparison,RankDataset
from zubr.Optimizer cimport RankOptimizer
from zubr.ZubrClass cimport ZubrLoggable
from zubr.Learner cimport OnlineLinearLearner
from zubr.Extractor cimport Extractor
from zubr.util.optimizer_util import find_data
import numpy as np
cimport numpy as np
from zubr.Optimizer import Optimizer 

cdef class FeatureSelector(ZubrLoggable):
    """Base feature selector class"""

    @classmethod
    def from_config(self,config):
        """Create a feature selector instance from config 

        :param config: the selector configuration 
        """
        raise NotImplementedError

cdef class GreedyWrapperSelector(FeatureSelector):
    """Base class for greedy feature selectors"""

    cdef int selection_loop(self) except -1:
        """Measure the models score

        :rtype: None 
        """
        raise NotImplementedError

    cpdef int select(self) except -1:
        """Main python method to start feature selection 

        :rtype: None 
        """
        raise NotImplementedError

cdef class ForwardSearch(GreedyWrapperSelector):
    pass

cdef class BackwardSearch(GreedyWrapperSelector):
    """Does backward feature selection, i.e., starts with a model and all features
    and greedily removes features that improve the validation score
    """

    def __init__(self,model,config):
        """Initializes a backward search instance


        :param model: the model to select features from 
        """
        self.model = <RankOptimizer>model
        self.config = config

        if not self.config.test_templates and not self.config.test_individual:
            raise ValueError('No selection method selected!')

    cpdef int select(self) except -1:
        """Starts the selection process

        :rtype: None
        """
    
        try:
            self.logger.info('Starting selection loop...')
            self.selection_loop()
        except Exception,e:
            #self.logger.error('Unexpected error during selection: %s' % e)
            self.logger.error(e, exc_info=True)
            #traceback.print_exc(file=sys.stdout)
            
    cdef int selection_loop(self) except -1:
        """Select features by removing some and testing on validation data 

        -- The selectionw works in three ways:
        
            -- 1: select features that cause the most problems on the validation
                 set weighted a combination of their frequeny and feature weight 
            -- 2: greedily go through each template and try to remove 
            -- 3: The combination of 2 and 3 (in the sequence) 

        :rtype: None 
        """
        cdef int i,j

        ## the underlying model and model parameters 
        cdef RankOptimizer optimizer = <RankOptimizer>self.model
        cdef Extractor extractor = <Extractor>optimizer.extractor
        cdef OnlineLinearLearner lmodel = <OnlineLinearLearner>optimizer.model
        cdef double[:] parameters = lmodel.w

        ## selection configuration 
        cdef object config = self.config
        
        ## extractor config
        cdef object econfig = extractor.config
        cdef object starts = econfig.tempmanager
        cdef long[:] templates = np.sort(starts.keys())
        cdef long num_features = econfig.num_features
        cdef int num_templates = templates.shape[0]
        
        ## backup features to test
        cdef double[:] backup
        cdef long start,end
        cdef int distance
        cdef int o

        ## selection type
        cdef bint remove_templates = config.test_templates
        cdef bint remove_individual = config.test_individual

        ## removed templates
        cdef list removed = []

        ## incorrect features and feature frequency 
        cdef dict candidates
        cdef int max_features = config.max_feat
        cdef long[:] to_prune
        cdef double backup_val
        
        ## scores
        cdef RankScorer new_score
        cdef double init_score = optimizer.vscore.score()
        
        ## validation dataset
        cdef RankDataset validation = find_data(optimizer.config,'valid')

        ## try greedily removing individual function
        
        if remove_individual:

            ## Find features that occur in wrong analyzes with frequency 
            candidates = wrong_features(config)
            self.logger.info('Trying to find feature to prune...')

            ## weight these using the model, and extract mac 
            to_prune = _find_candidates(candidates,max_features,parameters)
            self.logger.info('Found %d features, now testing' % max_features)

            ## test model without most frequent
            for i in range(max_features):
                backup_val = parameters[to_prune[i]]
                parameters[to_prune[i]] = 0.0
                new_score = optimizer._test_model(validation,'valid-select',it=to_prune[i],debug=False)

                ## check if the score increases
                if new_score > optimizer.vscore:
                    self.logger.info('Improvement achieved after removing feature %d' % to_prune[i])
                    optimizer.vscore = new_score

                ## restore feature value if not
                else:
                    parameters[to_prune[i]] = backup_val

        ## try to remove entire templates 
        if remove_templates:

            ## iteration through templates 
            for i in range(num_templates):
                end   = num_features if (i+1 == num_templates) else starts[templates[i+1]]
                start = starts[templates[i]]
                backup = np.zeros((end-start,),dtype='d')
                distance = end-start

                ## copy values and temporarily set to zero in model
                for j in range(0,distance):
                    backup[j] = parameters[start+j]
                    parameters[start+j] = 0.0

                ### evaluate without these features 
                new_score = optimizer._test_model(validation,'valid-select',it=templates[i],debug=False)

                ## has the score improved? 
                if new_score > optimizer.vscore:
                    self.logger.info('Improvement achieved after removing template %d' % templates[i])
                    optimizer.vscore = new_score
                    removed.append(templates[i])

                ## if not restore values 
                else:
                    ## restore the feature values
                    for j in range(0,distance):
                        parameters[start+j] = backup[j]

        ## backup model if improvement noticed
        #if <double>optimizer.vscore.score() > init_score:
        if config.eval_after:
            optimizer.test(test_type='test')

        if not config.pipeline_backup:
            backup_better(config,optimizer.dump_large,removed)

    @classmethod
    def from_config(cls,config):
        """Load a backward feature selector from configuration 

        :param configuration: the main configuration 
        :returns: a feature selector instance 
        """
        if not config.pipeline_backup:
            model = RankOptimizer.load_large(config.model_loc)
        else:
            oclass = Optimizer(config.optim)
            model = oclass.load_backup(config)

        return cls(model,config)

## find top features that appear in incorrect anaylsis weighted by model weight and frequency

cdef np.ndarray _find_candidates(dict incorrect, ## features to consider
                               int max_features, ## maximum number of features to keep
                               double[:] parameters):
    """Find the highest ranking features in incorrect analyses to consider pruning 

    :param incorrect: the incorrect features 
    :param max_features: the maximum features to consider pruning 
    """
    cdef long identifier
    cdef int count
    cdef long psize = parameters.shape[0] 
    cdef double[:] weighted_counts = np.zeros((psize,),dtype='d')
    
    for (identifier,count) in incorrect.iteritems():
        weighted_counts[identifier] += parameters[identifier]*float(count)

    return np.argsort(weighted_counts)[::-1][:max_features]

### CLI Stuff

def params():
    """Main feature selection parameters

    :rtype: tuple
    :returns: description of options with names, 
    """
    options = [
        ("--test_templates","test_templates","","str",
         "greedily try to eliminate templates [default='']","FeatureSelection"),
        ("--max_feat","max_feat",100,"int",
         "the maximum number of features to remove [default=False]","FeatureSelection"),
        ("--model_dir","model_dir","","str",
         "the directory with the model [default='']","FeatureSelection"),
        ("--test_templates","test_templates",False,"bool",
         "Greedily try to remove entire templates [default='']","FeatureSelection"),
        ("--test_individual","test_individual",False,"bool",
         "Greedily try to remove individual features  [default='']","FeatureSelection"),
         ("--eval_after","eval_after",False,"bool",
         "eval test after selection  [default=False]","FeatureSelection"),
    ]

    model_group = {"FeatureSelection": "settings for feature selection"}
    return (model_group,options)

def argparser():
    """Returns an aligner argument parser using default

    :rtype: zubr.util.config.ConfigObj 
    :returns: default argument parser 
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m feature_selector [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser


def main(argv):
    """The main point of execution for FeatureSelectors

    :param config: the main configuration fo running feature selectors
    """
    try:
        if isinstance(argv,ConfigAttrs):
            config = argv
        else:
            parser = argparser()
            config = parser.parse_args(argv[1:])
            logging.basicConfig(level=logging.DEBUG)
            ## restore config
            restore_config(config)

        ## selector instance
        selector = BackwardSearch.from_config(config)
        selector.select()

    except Exception,e:
        traceback.print_exc(file=sys.stdout)
