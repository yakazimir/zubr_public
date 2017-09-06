# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Base classe for all feature extractors 


To implement a new feature extractor, inherit from ExtractBase and 
add class to EXTRACTORS map in  FeatureExtractor.pyx 

"""

from zubr.ZubrClass cimport ZubrSerializable
from zubr.Features cimport FeatureObj
from zubr.Dataset cimport RankPair,RankComparison,RankDataset
import numpy as np
cimport numpy as np


cdef class ExtractorBase(ZubrSerializable):

    """A feature extractor base class

    methods to implement
    --------------------------

    build_extractor(..) : build the extractor from a zubr configuration 
    offline_init(..)    : called when first starting to use optimizer 


    attributes to implement 
    ---------------------------

    num_features : the space of features in extractor


    -- When building from a configuration, the configuration attribute 
    config.dir must be set, and all data should be given at this location


    IMPLEMENTING A NEW EXTRACTOR: 
    ==================================
    
    When implementing a new extractor to use with an optimizer in Optimizer, 
    you must inherit from Extractor

    """

    @classmethod
    def build_extractor(cls,config):
        """Build an feature extractor instance

        :param config: extractor configuration 
        :type config: zubr.util.config.ConfigAttrs
        :rtype: cls
        """
        raise NotImplementedError

    cpdef void offline_init(self,object dataset,str rtype):
        """Called when starting optimization, can be used for 
        initializing extractor in some way. By default, it is passed

        :param dataset: the dataset used with extractor 
        :param rtype: the type of data
        """
        pass

    cdef void after_eval(self,RankDataset dataset,RankComparison ranks):
        """Called after evaluation when new ranks have been computed, 
        passes by default 

        :param dataset: the dataset just evaluated 
        :param ranks: the computer ranks from evaluation 
        """
        pass

    cdef FeatureObj extract(self,RankPair instance,str etype):
        """Main method to call for extracting features
        
        :param instance: the data instance to extract from 
        :param etype: the extraction type (e.g., training,validation,...)
        :rtype FeatureObj
        """
        raise NotImplementedError

    cdef FeatureObj extract_query(self,RankPair instance,str etype):
        """Extract features for a given query during runtime 
        
        :param query: a query encoded in the underlying model representation
        :param surface: the surface form of the query 
        """
        raise NotImplementedError

    cdef RankComparison rank_init(self,str etype):
        """Called before evaluating model on a rank dataset

        :param etype: the evaluation type (e.g., train,test, ...)
        :returns: A rank comparison object 
        """
        raise NotImplementedError

    property num_features:
        """Returns the number of features using extractor
        
        -- such an attribute is used to build feature vectors
        """
        
        def __get__(self):
            """Returns and gets the number of extactor features 
            
            :rtype: int
            """
            raise NotImplementedError

    property dir:
        """Information about the directory where the extractor sits"""
        def __set__(self,new_dir):
            raise NotImplementedError
        def __get__(self):
            raise NotImplementedError

    property offset:
        """When using the extractor, it might happen that you need to set an ``offset`` e.g., 
        if you split up the data use the extractor individually for these different parts. This 
        is because the extractor might contain components that are indexed according to the original
        dataset order. Be careful with this!"""
        def __set__(self,new_offset):
            raise NotImplementedError
        def __get__(self):
            raise NotImplementedError

    ## enter and exit

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def exit(self):
        pass
    
cdef class Extractor(ExtractorBase):
    pass

cdef class RankExtractor(ExtractorBase):
    """A feature extractor for cases where the goal is to rank items"""
    pass
