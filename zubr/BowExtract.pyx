# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Feature extractor for bag of word models

"""
import time
import numpy as np
cimport numpy as np
from zubr.Extractor cimport Extractor
from zubr.Features cimport FeatureObj,FeatureMap,TemplateManager
from zubr.Dataset cimport RankDataset,RankPair,Dataset,RankStorage,RankComparison
from zubr.util.config import ConfigAttrs
from zubr.util.bow_util import build_extractor
from cython cimport boundscheck,wraparound,cdivision
from zubr.Phrases cimport PhrasePair,ConstructPair

cdef class BowExtractor(Extractor):
    """Object for extract bag of words features"""

    def __init__(self,config):
        """Initializes a bow feature extractor

        :param config: the extractor configuration
        """
        self._config = config
        self.trainranks = RankStorage.load_empty(0,0)
        self.testranks  = RankStorage.load_empty(0,0)
        self.validranks = RankStorage.load_empty(0,0)
                
    @classmethod
    def build_extractor(cls,config):
        """Builds an aligner extraction instance from config 

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        :rtype: BowExtractor
        """
        settings_and_data = ConfigAttrs()
        build_extractor(config,settings_and_data)

        return cls(settings_and_data)

    cpdef void offline_init(self,object dataset,str rtype):
        """Initialize the ranks

        -- one problem: all the ranks are the same, is 

        :param dataset: the dataset to use for initialization 
        :param rtype: the type of initialization 
        """
        cdef RankStorage storage
        cdef object config = self._config
        self.logger.info('Initializing ranks for <%s> data' % rtype )
        st = time.time()

        if rtype == 'train' and self.training_ranks.is_empty:
            storage = make_storage(dataset,config,rtype)
            storage.compute_score(config.dir,rtype)
            self.training_ranks = storage

        elif rtype == 'valid' and self.validation_ranks.is_empty:
            storage = make_storage(dataset,config,rtype)
            storage.compute_score(config.dir,rtype)
            self.validation_ranks = storage 

        elif rtype == 'test' and self.test_ranks.is_empty:
            storage = make_storage(dataset,config,rtype)
            storage.compute_score(config.dir,rtype)
            self.test_ranks = storage

        else:
            self.logger.info('Offline <%s> storage already computed...' % rtype)

        self.logger.info('Offline <%s> initialization finished in %f seconds' %\
                    (rtype,time.time()-st))

    cdef FeatureObj extract(self,RankPair instance,str etype):
        """Main method for extraction 

        :param instance: the english input to extract for
        :param etype: the type of data
        """
        cdef object config = self._config
        cdef bint from_file = config.store_feat
        cdef int identifier = instance.global_id
        cdef bint off = FeatureObj.features_exist(config.dir,etype,identifier)
        
        if from_file and off:
            return self.extract_from_file(config.dir,etype,identifier)
        return self.extract_from_scratch(instance,etype)

    cdef RankComparison rank_init(self,str etype):
        """Return a rank comparison instance for testing model on data


        :param etype: the type of data 
        :raises ValueError: if the etype if unknown 
        """
        cdef object config = self._config
        cdef int size = config.beam
        
        if etype == 'train':
            return RankComparison(self.training_ranks,size)
        elif etype == 'test':
            return RankComparison(self.test_ranks,size)
        elif etype == 'valid':
            return RankComparison(self.validation_ranks,size)
        raise ValueError('Etype unknown: %s' % etype)

    cdef FeatureObj extract_from_file(self,str directory,str etype, int identifier):
        """Extract features representations from a file
        
        -- Each line in the feature representation is a different entry

        :param directory: the target directory 
        :param etype: the type of feature representation (e.g., train,test,...)
        :param identifier: the example identifier
        """
        cdef object config = self._config
        return FeatureObj.load_from_file(config.dir,etype,identifier,
                                             config.num_features)

    cdef FeatureObj extract_from_scratch(self,RankPair instance,str etype):
        """Do feature extract for a given example from scratch

        :param instance: the english input to extract features for 
        :param etype: the type of data
        """
        cdef object config = self._config
        cdef bint from_file = config.store_feat
        cdef RankStorage ranks
        cdef int[:] rank_items
        cdef int identifier = instance.global_id
        cdef int beam = config.beam,rank_size
        cdef int gold_pos = -1,gold_item = -1
        cdef dict temps = config.tempmanager
        cdef long num_features = config.num_features
        cdef FeatureObj features
        cdef int value_pos
        cdef unicode surface = instance.surface

        ## english input 
        cdef int[:] input_v = instance.en
        
        if etype == 'train':
            ranks = self.training_ranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]
            gold_pos = ranks.gold_position(identifier)

        elif etype == 'valid':
            ranks = self.validation_ranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]
            
        elif etype == 'test':
            ranks = self.test_ranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]
        
        features = FeatureObj(rank_size,templates=temps,maximum=num_features)
            
        if gold_pos > -1:
            gold_item = ranks.gold_value(identifier)
            main_extractor(config,input_v,gold_item,surface,
                               features.gold_features)
            ## extract

        for value_pos in range(rank_size):
            new_identifier = rank_items[value_pos]
            if gold_pos > -1: assert new_identifier != gold_item,"Beam has gold!"
            ## extract
            main_extractor(config,input_v,new_identifier,surface,features[value_pos])
            
        if from_file:
            features.print_features(config.dir,etype,identifier,
                                        rank_items,gold_item)

        return features
        
    property num_features:
        """Returns the number of features"""

        def __get__(self):
            """Returns and get the number of extractor features

            :rtype: long 
            :returns: the number of features
            """
            return self._config.num_features

    property training_ranks:
        """Stores the ranks for the training data"""

        def __get__(self):
            return <RankStorage>self.trainranks
        
        def __set__(self,RankStorage new_ranks):
            self.trainranks = new_ranks

    property test_ranks:
        """Stores the ranks for the testing data"""

        def __get__(self):
            return <RankStorage>self.testranks
        
        def __set__(self,RankStorage new_ranks):
            self.testranks = new_ranks            

    property validation_ranks:
        """Stores the ranks for the validation data (if provided)"""

        def __get__(self):
            return <RankStorage>self.validranks
        
        def __set__(self,RankStorage new_ranks):
            self.validranks = new_ranks
        
    def __reduce__(self):
        ## pickle implementation
        return BowExtractor,(self._config,)


##c level methods

cdef RankStorage make_storage(RankDataset dataset,object config,str dtype):
    """Create a storage item for a given dataset 

    -- issue: it storage an individual copy of the rank list for each 
    item even though the rank item is constant (a bit silly)

    :param dataset: the dataset to find storage for 
    :param config: the extractor configuration 
    :param dtype: the type of data to create for 
    """
    cdef np.ndarray rank_list = config.ranks
    cdef int rlen = rank_list.shape[0]
    cdef int i,j,size = dataset.size
    cdef int[:] gold_ranks = dataset.rank_index
    cdef RankStorage storage = RankStorage.load_empty(size,rlen)

    ## storage information 
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos

    ## this is a bit silly, but let's keep it for now
    for i in range(size):
        gold_pos[i] = gold_ranks[i]
        for j in range(rlen): ranks[i][j] = j
    return storage

cdef inline long word_index(int eid,int fid,int flen):
    """find the feature number for word id

    :param eid: the english word identifier 
    :param fid: the foreign word identifier
    :param flen: the total size of f words
    :returns: a unique identifier for pair
    """
    if (eid == -1 or fid == -1):
        return -1
    return (eid*flen)+fid

cdef inline void __word_feat(PhrasePair pair,## the unigram pair
                         int eid,int fid,long pair_id, ## input ids
                         bint stop,bint ignore_stop, ## options
                         FeatureMap features, ## feature representation
                         ):
    """The main bag of word features"""
    #if ignore_stop and stop: return
    
    ## skips over pairs if one word in unknown 
    if pair_id > -1:
        features.add_binary(0,pair_id)
        features.add_binary(1,eid)
        features.add_binary(2,fid)

    ## match features?

cdef void main_extractor(object config,
                             int[:] input_v,int output_id, ## input/output
                             unicode english, ## english surface form
                             FeatureMap features):
    """The main extractor method 

    :param config: the extractor configuration 
    :param input_v: the english input vector
    :param output_id: the id of the foreign output 
    :param english: the english surface form 
    :param features: (empty) the feature representaition to create
    """
    cdef str lang = config.lang
    cdef bint ignore_stops = config.ignore_stops
    cdef bint match_feat = config.match_feat    

    cdef np.ndarray rank_list = config.ranks
    cdef np.ndarray rank_surface = config.rank_vals

    ## output representation
    cdef int[:] output_v = rank_list[output_id]
    cdef unicode foreign = np.unicode(rank_surface[output_id])

    ## relative lengths 
    cdef int elen = input_v.shape[0]
    cdef int flen = output_v.shape[0]
    cdef int eid,fid
    cdef long pair_id
    cdef int fvlen = config.flen 
    cdef int i,j

    ## original inputs
    cdef list elist = english.split()
    cdef list flist = [u"<unknown>"]+foreign.split()

    ## pair
    cdef PhrasePair word_component
    cdef bint is_stop
    cdef int matches = 0

    for i in range(elen):
        eid = input_v[i]
        eword = elist[i]

        for j in range(1,flen):
            fid = output_v[j]
            fword = flist[j]
            pair_id = word_index(eid,fid,fvlen)
            word_component = ConstructPair(eword,fword,lang=lang)
            is_stop = word_component.is_stop_word()
            ## number of matches
            if word_component.sides_match(): matches += 1
            
            ## word-level features
            __word_feat(word_component,
                            eid,fid,pair_id, # ids
                            is_stop,ignore_stops, ## stop settings 
                            features)

    ## has no features? add zero feature
    if features.feature_size == 0:
        features.add_binary(0,0)
            
## CLI stuff

def params():
    """Defines the different parameters for the aligner extractor

    :rtype: tuple 
    :returns: description of switches and default settings
    """
    options = [
        ("--bow_data","bow_data",'',"str",
             "the location of the data [default='']","BOW"),
        ("--store_feat","store_feat",False,"bool",
             "extract features only once to file  [default=False]","BOW"),
        ("--ignore_stops","ignore_stops",False,"bool",
             "ignore input stop words [default=False]","BOW"),
        ("--lang","lang",'en',"str",
              "the language of the reranker [default='en']","BOW"),
        ("--templates","templates","pair+indv","str",
             "extract pair ids, with or without individiual [default='pair+indv']","BOW")
    ]
        
    extractor_group = {"BOW":"BOW feature extractor"}
    return (extractor_group,options)
