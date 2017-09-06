# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Feature components, often container classes, for holding different 
types of information needed for doing feature extraction 

"""
import time
import sys
import os
import re
from zubr.ZubrClass cimport ZubrSerializable
import numpy as np
cimport numpy as np
from zubr.Phrases cimport HieroPhraseTable,SimplePhraseTable,SparseDictWordPairs,DescriptionWordPairs
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Dataset cimport RankStorage

cdef class WordPhraseComponents(ZubrSerializable):
    """Class for storing phrase related features components"""

    def __init__(self,pairs,phrases,hiero):
        """Create a phrase component instance 

        :param pairs: the pair lookup container 
        :param phrases: the phrase lookup container 
        :param hiero: the hierarchical phrase rule lookup container
        """
        self.pairs   = pairs
        self.phrases = phrases
        self.hiero   = hiero

    def backup(self,wdir):
        """Backup the word phrase components 

        :param wdir: the working directory 
        :rtype: None 
        """
        self.logger.info('Backing up word/phrase components...')        
        self.pairs.backup(wdir)
        self.phrases.backup(wdir)
        self.hiero.backup(wdir)

    @classmethod
    def load_backup(cls,config):
        """Load a backup and create instance 

        :param config: the main configuration 
        """
        pairs = SparseDictWordPairs.load_backup(config)
        phrases = SimplePhraseTable.load_backup(config)
        hiero = HieroPhraseTable.load_backup(config)
        return cls(pairs,phrases,hiero)

cdef class KnowledgeComponents(ZubrSerializable):
    """Holds items related to background knowledge features """
    def __init__(self,descriptions):
        self.descriptions = descriptions

    def backup(self,wdir):
        """Backup the knowledge components to file 
        
        :param wdir: the working directory 
        :rtype: None 
        """
        self.descriptions.backup(wdir)

    @classmethod
    def load_backup(cls,config):
        """Load the backup item 

        :param config: the extractor configuration 
        """
        stime = time.time()
        descriptions = DescriptionWordPairs.load_backup(config)
        instance =  cls(descriptions)
        instance.logger.info('Load in %s seconds' % str(time.time()-stime))
        return instance


def __find_langs(rank_vals,lang_map):
    """Determines if there are languages here 

    """
    for item in rank_vals:
        first = item.split()[0]
        if not "<!" in first and not ">!" in first: continue
        if first not in lang_map:
            lang_map[first] = len(lang_map)


cdef class RankComponents(ZubrSerializable):
    """Holds items related to the ranks"""

    def __init__(self,rank_list,rank_vals,tree_list,classes):
        """Initialize a rank components item list 

        Note: it will try to identify if multiple language are present 
        here by looking through the rank list and find representations 
        that start with <!name_of_lang!>, and will store this is a map. 

        """
        self.rank_list = rank_list
        self.rank_vals = rank_vals
        self.trees     = tree_list
        self.classes   = classes
        self.langs     = {}

        ## go through to find languages
        __find_langs(rank_vals,self.langs)
        self.logger.info('Number of rank output languages: %d' % len(self.langs))
                                    
    cdef int[:] rank_item(self,int index):
        """Retieve a rank item for feature extraction 

        :param index: the index of the rank item 
        """
        raise NotImplementedError

    cdef int language_id(self,int index):
        """Returns the language associated with """
        raise NotImplementedError

    cdef unicode surface_item(self,int index):
        """Retieve the surface representation for the rank item 

        :param index: the index of the rank item 
        """
        raise NotImplementedError
    
    def backup(self,wdir):
        """Backup the rank components 

        :param wdir: the working directory 
        :rtype: None 
        """
        ndir = os.path.join(wdir,"rank_components")
        if os.path.isdir(ndir):
            self.logger.info('Already backed up, skipping...')
            return

        ## make new directory
        stime = time.time()
        os.mkdir(ndir)

        ## rank list backup
        rank_out = os.path.join(ndir,"rank_items")
        np.savez_compressed(rank_out,self.rank_list,self.rank_vals,self.trees,self.classes)
        
        ## log time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load a given backup and create an instance 

        :param config: the global configuration object
        :returns: rank component instance 
        """
        ndir = os.path.join(config.dir,"rank_components")
        stime = time.time()

        ## load the components
        components = os.path.join(ndir,'rank_items.npz')
        archive = np.load(components)

        ## archive 
        rank_list = archive["arr_0"]
        rank_vals = archive["arr_1"]
        trees     = archive["arr_2"]
        classes   = archive["arr_3"].item()
        
        instance = cls(rank_list,rank_vals,trees,classes)
        instance.logger.info('Loaded backup in %s seconds' % (time.time()-stime))
        return instance

    ## properties
    property num_langs:
        """The number of type of languages included in the rank list"""
        def __get__(self):
            """ 

            :rtype: int 
            """
            return len(self.langs)
        
cdef class PolyComponents(RankComponents):
    """Components for polyglot models """

    cdef int[:] rank_item(self,int index):
        """Retieve a rank item for feature extraction 

        Note : with the polyglot representations, the strting point 
        has a language identifier, which we don't need for the reranker. 

        Therefore, this function returns the representation without this 
        initial langugage identifier 

        :param index: the index of the rank item 
        """
        cdef np.ndarray rank_list = self.rank_list
        return np.insert(rank_list[index][2:],0,0)

    cdef unicode surface_item(self,int index):
        """Retieve the surface representation for the rank item 

        :param index: the index of the rank item 
        """
        cdef np.ndarray rank_vals = self.rank_vals
        return np.unicode(rank_vals[index].split()[1:])

    cdef int language_id(self,int index):
        cdef dict lmap = self.langs
        cdef np.ndarray rank_vals = self.rank_vals
        first = rank_vals[index].split()[0]
        return lmap.get(first,-1)

cdef class MonoComponents(RankComponents):
    """Components for monolingual data"""

    cdef int[:] rank_item(self,int index):
        """Retieve a rank item for feature extraction 

        :param index: the index of the rank item 
        """
        cdef np.ndarray rank_list = self.rank_list
        return rank_list[index]

    cdef int language_id(self,int index):
        return -1

    cdef unicode surface_item(self,int index):
        """Retieve the surface representation for the rank item 

        :param index: the index of the rank item 
        """
        cdef np.ndarray rank_vals = self.rank_vals
        return np.unicode(rank_vals[index])

cdef class StorageComponents(ZubrSerializable):
    """A container class for various storages items for the reranker """

    def __init__(self,train,valid,test,query):
        """Create a StorageComponents instance 

        :param train: train rank storage 
        :param valid: validation rank storage 
        :param test: test rank storage 
        :param query: query rank storage 
        """
        self.trainranks = train
        self.validranks = valid
        self.testranks  = test
        self.queryranks = query
        
    def backup(self,wdir):
        """Backup the storage items to file 

        :param wdir: the experiment working directory 
        :rtype: None 
        """
        self.logger.info('Backing up the rank storage containers...')
        stime = time.time()

        ## backup each individual item 
        self.trainranks.backup(wdir,name='train')
        self.testranks.backup(wdir,name='test')
        self.validranks.backup(wdir,name='valid')
        self.queryranks.backup(wdir,name='query')

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load a given backup from file and configuration 

        :param config: the global configuration object 
        """
        stime = time.time()
        train = RankStorage.load_backup(config,name='train')
        test  = RankStorage.load_backup(config,name='test')
        valid = RankStorage.load_backup(config,name='valid')
        query = RankStorage.load_backup(config,name='query')
        
        ## create instance
        instance = cls(train,valid,test,query)
        instance.logger.info('Loaded instance from file in %s seconds' % str(time.time()-stime))
        return instance 

