# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Feature extractor for aligners

"""
import sys
import time
import numpy as np
cimport numpy as np
from zubr.Extractor cimport Extractor
from zubr.Features cimport FeatureObj,FeatureMap,TemplateManager
from zubr.Dataset cimport RankDataset,RankPair,Dataset,RankStorage,RankComparison
from zubr.util.config import ConfigAttrs
from zubr.util.align_extractor import build_extractor,find_base
#from zubr.Aligner cimport AlignerBase
#from zubr.SymmetricAligner cimport SAlign,SymAlign,Phrases

## new stuff
from zubr.Alignment cimport WordModel
from zubr.SymmetricAlignment cimport SymmetricWordModel,SymAlign,Phrases

from zubr.Phrases cimport PhrasePair,ConstructPair,ConstructHPair,HieroRule
#from zubr.Aligner cimport AlignerBase
from cython cimport boundscheck,wraparound,cdivision

cdef class AlignerExtractor(Extractor):
    """Object for extracting features from symmetric alignments"""

    def __init__(self,base,config):
        """ 

        :param base: the base alignment model
        :param config: the main configuration 
        """
        self.base_model = base
        self._config    = config
        self.trainranks = RankStorage.load_empty(0,0)
        self.testranks  = RankStorage.load_empty(0,0)
        self.validranks = RankStorage.load_empty(0,0)
        self.queryranks = RankStorage.load_empty(0,0)

    @classmethod
    def build_extractor(cls,config):
        """Builds an aligner extraction instance from config 

        -- a lot of the dirty work is done in zubr.util.

        :param config: extractor and experiment configuration 
        :type config: zubr.util.config.ConfigAttrs
        :rtype: AlignerExtractor
        """
        #base = AlignerBase.from_model(find_base(config))
        base = SymmetricWordModel.load(find_base(config))
        settings_and_data = ConfigAttrs()
        build_extractor(base,config,settings_and_data)

        return cls(base,settings_and_data)

    cdef RankComparison rank_init(self,str etype):
        """Returns a rank comparison instance for evaluation

        :param etype: the type of data (e.g., train/test/valid/...)
        :raises ValueError: when the etype if unknown
        """
        cdef object config = self._config
        cdef int beam = config.beam
        
        if etype == 'train':
            return RankComparison(self.training_ranks,beam)

        elif etype == 'train-test':
            return RankComparison(self.training_ranks,beam)
        
        elif etype == 'test':
            return RankComparison(self.test_ranks,beam)

        elif etype == 'valid' or etype == 'valid-select':
            return RankComparison(self.validation_ranks,beam)

        elif etype == 'query':
            return RankComparison(self.query_ranks,beam)

        raise ValueError('Etype unknown: %s' % etype)

    cpdef void offline_init(self,object dataset,str rtype):
        """Build the ranks for the various datasets (i.e., train/test/validation) 

        -- Idea: rather than ranking the target components each time with the aligner, 
        the ranks for each item are stored for resuse each training cycle (to reduce time).

        :param dataset: the dataset to use for initialization 
        :param rtype: the type of initialization (e.g., training/validation/testing..)
        :rtype: None
        """
        cdef RankStorage storage
        cdef object config = self._config
        if rtype != 'valid-select' and rtype != 'query':
            self.logger.info('Initializing ranks for <%s> data' % rtype)
        st = time.time()

        #if rtype == 'train' and <bint>self.training_ranks.is_empty:
        if 'train' in rtype and <bint>self.training_ranks.is_empty:
            #storage = rank_with_basemodel(dataset,<SAlign>self.base_model,config,rtype)
            storage = rank_with_basemodel(dataset,<SymmetricWordModel>self.base_model,config,rtype)
            storage.compute_score(config.dir,rtype)
            self.training_ranks = storage

        elif (rtype == 'valid' or rtype == 'valid-select') and <bint>self.validation_ranks.is_empty:
            #storage = rank_with_basemodel(dataset,<SAlign>self.base_model,config,rtype)
            storage = rank_with_basemodel(dataset,<SymmetricWordModel>self.base_model,config,rtype)
            storage.compute_score(config.dir,rtype)
            self.validation_ranks = storage

        elif rtype == 'test' and <bint>self.test_ranks.is_empty:
            #storage = rank_with_basemodel(dataset,<SAlign>self.base_model,config,rtype)
            storage = rank_with_basemodel(dataset,<SymmetricWordModel>self.base_model,config,rtype)
            storage.compute_score(config.dir,rtype)
            self.test_ranks = storage

        ## query
        elif rtype == 'query':
            storage = rank_with_basemodel(dataset,<SymmetricWordModel>self.base_model,config,rtype)
            self.query_ranks = storage
            return

        ## need to add something for evaluating on training training
        #elif rtype == 'train-test':

        else:
            if rtype != 'valid-select':
                self.logger.info('Offline <%s> ranks already computed..' % rtype)
            return

        self.logger.info('Offline <%s> initialization finished in %f seconds' %\
                    (rtype,time.time()-st))

    cdef FeatureObj extract_query(self,RankPair instance,str etype):
        """Extract in a non-offline fashion features for a given query

        :param query: an encoded query 
        """
        return self.extract_from_scratch(instance,etype)
    
                            
    cdef FeatureObj extract(self,RankPair instance,str etype):
        """Main method for extracting feature for a given instance

        :param instance: the english input to extract features for 
        :param etype: the extraction scenario (e.g., train/test,...)
        :rtype: FeatureObj
        """
        cdef object config = self._config
        cdef bint from_file = config.store_feat
        cdef int identifier = instance.global_id
        cdef bint off = FeatureObj.features_exist(config.dir,etype,identifier)

        if from_file and off:
            return self.extract_from_file(config.dir,etype,identifier)
        return self.extract_from_scratch(instance,etype)
    
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
        """Do feature extraction from scratch

        :param instance: the english input to extract features for 
        :param etype: the type of extraction (e.g., training,text,...)
        :rtype: FeatureObj
        :returns: a feature representation of data
        """
        cdef RankStorage ranks
        cdef object config = self._config
        cdef bint from_file = config.store_feat
        cdef FeatureObj features
        cdef int identifier = instance.global_id
        cdef unicode surface = instance.surface
        cdef int[:] input_v = instance.en
        cdef int gold_id = instance.rep_id
        cdef int[:] ranks_items
        cdef int beam = config.beam
        cdef int rank_size
        cdef int gold_pos = -1,gold_item = -1
        cdef int value_pos
        #cdef SAlign base = <SAlign>self.base_model
        cdef SymmetricWordModel base = <SymmetricWordModel>self.base_model
        cdef dict temps = config.tempmanager
        cdef long num_features = config.num_features
        cdef int new_identifier

        if <bint>(etype == 'train'):
            ranks = <RankStorage>self.training_ranks
            #rank_items = ranks.find_ranks(identifier,etype,beam-1)
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]
            gold_pos = ranks.gold_position(identifier)
        elif <bint>(etype == 'train-test'):
            ranks = <RankStorage>self.training_ranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]

        elif <bint>(etype == 'valid'):
            ranks = <RankStorage>self.validation_ranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]

        elif <bint>(etype == 'test'):
            ranks = <RankStorage>self.test_ranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]

        elif etype == 'query':
            ranks = <RankStorage>self.query_ranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]

        else:
            raise ValueError('Uknown etype: %s' % etype)

        features = FeatureObj(rank_size,templates=temps,maximum=num_features)

        ## extract gold features (if training set if here)
        if gold_pos > -1:
            gold_item = ranks.gold_value(identifier)
            try: 
                main_extractor(base,config,input_v,surface,gold_item,gold_pos,
                                <FeatureMap>features.gold_features)
            except Exception,e:
                self.logger.error(e,exc_info=True)
                sys.exit('Extraction error encountered, check log')

        # ## extract the rest (in any case)
        for value_pos in range(rank_size):
            new_identifier = rank_items[value_pos]
            
            ## double check that rank value doesn't equal the gold value
            if gold_pos > -1:
                assert new_identifier != gold_item,"Beam contains gold!"
                
            try: 
                main_extractor(base,config,input_v,surface,new_identifier,value_pos,
                                <FeatureMap>features[value_pos])
            except Exception,e:
                self.logger.error(e,exc_info=True)
                sys.exit('Extraction error encountered, check log')
            
        ## write features to file for later use

        if from_file and etype != 'train-test' and etype != 'query':
            features.print_features(config.dir,etype,identifier,
                                        rank_items,gold_item)
        return features

    property num_features:
        """Returns the number of features using extractor"""
        def __get__(self):
            return self._config.num_features

    property training_ranks:
        """Stores the ranks for the training data"""

        def __get__(self):
            return <RankStorage>self.trainranks
        
        def __set__(self,RankStorage new_ranks):
            self.trainranks = new_ranks

    property query_ranks:
        """Stores the ranks for the query data"""

        def __get__(self):
            return <RankStorage>self.queryranks
        
        def __set__(self,RankStorage new_ranks):
            self.queryranks = new_ranks

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

    property config:
        """Stores the ranks for the validation data (if provided)"""

        def __get__(self):
            return self._config
        
    def __reduce__(self):
        ## pickle implementation
        #return AlignerExtractor,(self.base_model,self._config)
        return (rebuild_extractor,
                    (self.base_model,self._config,self.trainranks,
                         self.testranks,self.validranks))
        

def rebuild_extractor(base,config,trainr,testr,validr):
    extractor = AlignerExtractor(base,config)
    extractor.training_ranks = trainr
    extractor.validation_ranks = validr
    extractor.test_ranks = testr
    return extractor

## c level functions

@boundscheck(False)
@wraparound(False)
@cdivision(True)
#cdef RankStorage rank_with_basemodel(RankDataset dataset,SAlign base,object config,str dtype):
cdef RankStorage rank_with_basemodel(RankDataset dataset,SymmetricWordModel base,object config,str dtype):
    """Rank items with base aligner model

    :param dataset: the dataset to rank
    :param config: the main configuration 
    :param dtype: the type of data to rank (e.g., train/test,...) 
    """
    cdef int i,k,size = dataset.size
    cdef np.ndarray en = dataset.en
    cdef int[:] gold_ranks = dataset.rank_index
    cdef np.ndarray rank_list = config.ranks
    cdef int rlen = rank_list.shape[0]
    #cdef RankStorage storage = RankStorage(size,rlen)
    cdef RankStorage storage = RankStorage.load_empty(size,rlen)
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos
    #cdef AlignerBase ftoe = <AlignerBase>base.ftoe
    cdef WordModel ftoe = <WordModel>base.ftoe

    # ## run the model
    for i in range(size):
        #ftoe._rank(en[i],rank_list,ranks[i])
        ftoe._rank(en[i],rank_list,ranks[i])
        for j in range(rlen):
            if ranks[i][j] == gold_ranks[i]:
                gold_pos[i] = j

    return storage

## rank a single example



cdef unicode to_unicode(s):
    if isinstance(s,bytes):
        return (<bytes>s).decode('utf-8')
    return s

## auxiliary functions

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

cdef inline long cat_index(int eid,int aid,int alen):
    """find the feature number for word id

    :param eid: the english identifier 
    :param aid: the abstract class identifier 
    :param alen: the number of abstract classes
    """
    if eid == -1:
        return -1
    return (eid*alen)+aid

cdef inline long abstract_index(int aid,int eid,int elen):
    """Generate an identifier for an abstract symbol and english word
    
    :param aid: the abstract symbol id 
    :param eid: the english unigram id
    :param elen: the number of the english words or vocabulary size
    """
    if aid == -1 or eid == -1:
        return -1
    return (aid*elen)+eid

cdef inline void __len_feat(int elen,int flen, FeatureMap features):
    """Encodes information about input/output lengths

    :param elen: the length of the english input 
    :param flen: the length of the foreign output 
    :param features: the current features 
    """
    #features.add_binary(85,elen)
    #features.add_binary(86,flen)
    # if elen < 5: features.add_binary(87,0)
    # if elen > 5: features.add_binary(87,1)
    # if elen > 10: features.add_binary(87,2)
    # if elen > 15: features.add_binary(87,3)
    # if elen > 20: features.add_binary(87,4)
    if flen < 5:  features.add_binary(88,0)
    if flen > 5:  features.add_binary(88,1)
    if flen > 10: features.add_binary(88,2)
    if flen > 15: features.add_binary(88,3)
    if flen > 20: features.add_binary(88,4)
        
cdef inline void __relative_len(int elen,int fid,int classid, ## ids
                                    bint old_model,
                                    FeatureMap features):
    """Computes a feature for the length of the english input given foreign word

    -- Note: ignores is the fid is -1 (unknown word)

    :param elen: the english length
    :param fid: the foreign word identifier 
    :param flen: the size of the foreign vocabular 
    :param features: the current feature map (to add to)
    """
    cdef int start_val = (fid*10)
    cdef int base_id

    ## general feature about length
    if fid != -1:
        ## finer grained 
        if elen < 5: features.add_binary(14,start_val+0)
        elif 5 <= elen < 10: features.add_binary(14,start_val+1)
        elif 10 <= elen < 15: features.add_binary(14,start_val+2)
        elif 15 <= elen < 20: features.add_binary(14,start_val+3)
        elif 20 <= elen < 25: features.add_binary(14,start_val+4)
        elif 25 <= elen < 30: features.add_binary(14,start_val+5)
        elif 30 <= elen < 35: features.add_binary(14,start_val+6)
        elif elen >= 25: features.add_binary(14,start_val+7)
        ## less fine grained
        if elen < 10: features.add_binary(14,start_val+8)
        elif elen > 10: features.add_binary(14,start_val+9)

        ## class id relative 
        if classid != -1 and old_model:
            base_id = (classid*10)
            if elen < 5: features.add_binary(46,base_id+0)
            elif 5 <= elen < 10: features.add_binary(46,base_id+1)
            elif 10 <= elen < 15: features.add_binary(46,base_id+2)
            elif 15 <= elen < 20: features.add_binary(46,base_id+3)
            elif 20 <= elen < 25: features.add_binary(46,base_id+4)
            elif 25 <= elen < 30: features.add_binary(46,base_id+5)
            elif 30 <= elen < 35: features.add_binary(46,base_id+6)
            elif elen >= 25: features.add_binary(46,base_id+7)
            ## less fine grained
            if elen < 10: features.add_binary(46,base_id+8)
            elif elen > 10: features.add_binary(46,base_id+9)

cdef inline void __rank_features(int pos,FeatureMap features):
    """Assign features related to position rank


    :param pos: the position of the item in rank 
    :param features: the current features
    :rtype: None
    """
    if pos == 0:   features.add_binary(0,0)
    elif pos == 1: features.add_binary(0,1)
    elif pos == 2: features.add_binary(0,2)
    elif pos == 3: features.add_binary(0,3)
    elif pos == 4: features.add_binary(0,4)
    elif pos == 5: features.add_binary(0,5)
    elif pos == 6: features.add_binary(0,6)
    elif pos == 7: features.add_binary(0,7)
    elif pos == 8: features.add_binary(0,8)
    elif pos == 9: features.add_binary(0,9)        
    ## rank ranges
    if pos < 5:  features.add_binary(0,10)
    if pos < 10: features.add_binary(0,11)
    if 10 <= pos < 20: features.add_binary(0,12)
    if 20 <= pos < 30: features.add_binary(0,13)
    if pos >= 30: features.add_binary(0,14)
    ## 0 or not
    
cdef inline void __knowledge_features(int in_description,int dpid,int class_id,long cpid,int aid,int apid,int eid,int fid,
                                          #identifier
                                          bint match,int treepos,bint stop,bint in_alignment, ## features values
                                          FeatureMap features):
    """Word-level features related to background knowledge 

    :param in_description: word is in description 
    :param dpid: the id of word/description pair
    :param class_id: identifier of class (if exists)
    :param cpid: the word/description pair id
    :param aid: abstract class id of foreign word (-1 if not there)
    :param apid: the identifier of the abstract symbol/unigram pair
    :param eid: the english unigram identifier (-1 if unknown)

    :param match: the pairs match 
    :param stop: english side is a stop word 
    :param in_alignment: pair is current aligned (bool)
    :param features: the current feature map 
    """
    cdef bint description = True if in_description == 1 else False

    ## pair is in a description 
    if description:
        features.add_binned(34)
        if dpid > -1:  features.add_binary(33,dpid)
        features.add_binary(38,treepos)
        if match: features.add_binned(41)
        if in_alignment: features.add_binned(39)
        if eid > -1 and not stop: features.add_binary(77,eid)
        if fid > -1 and not stop: features.add_binary(78,fid)

    ## abstract class and content word 
    if class_id > -1 and not stop:
        features.add_binary(36,cpid)
        if match: features.add_binned(37)

    ## abstract symbol
    if aid > -1 and apid > -1 and not stop:
        features.add_binary(35,apid)
        features.add_binary(76,aid)
        features.add_binary(79,treepos)
        #if in_alignment and apid > -1: features.add_binary(81,apid)

cdef inline unicode __create_e_bigram(int index,list word_list):
    """"Create an english side bigram (if possible)

    :param index: the current word index
    :param word_list: the english word list 
    :returns: an empty string if index is okej, or bigram from index-1 -> index
    """
    if index == 0: return u''
    return ''.join(word_list[index-1:index+1])

cdef inline unicode __create_f_bigram(int index,list word_list):
    """Create an foreign side bigram (if possible)

    :param index: the current word index
    :param word_list: the english word list 
    :returns: an empty string if index is okej, or bigram from index-1 -> index
    """
    if index <= 1: return u''
    #return ' '.join(word_list[index-1:index+1])
    return to_unicode(''.join(word_list[index-1:index+1])) ## removes the spacing

cdef inline void __bigram_features(unicode english,unicode foreign,unicode ebigram, unicode fbigram,int treepos,
                                       int prevtree,bint prevdescr,int fid,FeatureMap features):
    """Computes the various bigram features, such as overlap, match, etc... and adds to feature map.

    :param english: the english current word
    :param foreign: the foreign current word
    :param ebigram: the english bigram (might be empty)
    :param fbigram: the foreign bigram (might be empty) 
    :param treepos: the tree position of the component word
    :param prevtree: the previous tree position
    :param prevdescr: the previous word in bigram is also in a description
    :param fid: the id of the foreign unigram

    :param features: the current features
    :rtype: None
    """

    cdef PhrasePair bigram_pair,eword_fbigram,ebigram_fword

    ## contiguous bigram pair both in descriptions of foreign word
    if ebigram and foreign and prevdescr:
        features.add_binned(53)
        if fid > -1: features.add_binary(52,fid)

    ## matching of bigrams 
    if ebigram and fbigram and english != foreign:
        bigram_pair   = ConstructPair(ebigram,fbigram)
        eword_fbigram = ConstructPair(english,fbigram)
        ebigram_fword = ConstructPair(ebigram,foreign)
        if prevtree == treepos:
            features.add_binary(54,treepos)
        
        ## bigram match and containment
        if bigram_pair.sides_match():
            features.add_binned(6)
            if prevtree == treepos: features.add_binary(51,treepos)
                
        elif bigram_pair.econtainsf():
            features.add_binned(7)
            features.add_binned(8)
            if prevtree == treepos: features.add_binary(51,treepos)
                
        elif bigram_pair.fcontainse():
            features.add_binned(7)
            features.add_binned(9)
            if prevtree == treepos: features.add_binary(51,treepos)
            
        ## e bigram and f unigram
        elif ebigram_fword.sides_match():
            features.add_binned(49)
            features.add_binned(47)
        elif eword_fbigram.sides_match():
            features.add_binned(49)
            features.add_binned(48)

cdef inline void __word_feat(PhrasePair pair,bint match,
                                 int eid,int fid,long pair_id, ## identifiers
                                 bint stop,int treepos,bint in_alignment, ## properties of pair 
                                 FeatureMap features):
    """Computes word match features, and composite features is they are specified

    :param pair: the pair of strings 
    :param match: the pair matches 
    :param pair_id: the pair id (-1 if one or more of the words are oov)
    :param stop: the english is a stop word 
    :param treepos: the tree position of the component
    :param in_alignment: appears in viterbi alignment
    """
    cdef bint oov = True if pair_id == -1 else False 

    ## english and foreign words
        
    ## add non-stop word pairs (might shut off depending on performance)
    if not oov and not stop:
        features.add_binary(2,pair_id)

    ## word identies
    if not oov and not stop:
        features.add_binary(83,eid)
        features.add_binary(84,fid)
    
    ## viterbi alignment
    if in_alignment and not stop:
        features.add_binary(11,treepos)
        if not oov:  features.add_binary(10,pair_id)
        if match: features.add_binned(55)

    ## matching content words 
    #if match and not stop:
    if match:
        features.add_binned(3)
        features.add_binary(5,treepos)
        if not oov: features.add_binary(4,pair_id)

    ## overlap
    elif pair.econtainsf() or pair.fcontainse() and not stop:
        features.add_binned(1)
        features.add_binary(50,treepos)

cdef void __compute_bins(FeatureMap features):
    """Computes the binned features

    :param features: the feature map (which contains the bins)
    :rtype: None 
    """
    features.compute_neg_bins(np.array([1,3,6,7,8,9,41,47,48,49,53,55,17,18,82,29,91,24,25,26,96], ## feature list 
                                           dtype=np.int32),2.0)
    
    features.compute_neg_bins(np.array([34,39,16,67,23,92], ## feature list 
                                           dtype=np.int32),4.0)

@boundscheck(False)
@wraparound(False)
@cdivision(True)
# cdef int main_extractor(SAlign base,object config,int[:] input_v,
#                              unicode english,int output_id, int pos,FeatureMap features) except -1:
cdef int main_extractor(SymmetricWordModel base,object config,int[:] input_v,
                             unicode english,int output_id, int pos,FeatureMap features) except -1:
    """The main extractor c method

    :param base: the base alignment model 
    :param config: the main extactor configuration with data
    :param input_v: the english input vector 
    :param english: the english str representation 
    :param output_id: the identifier of the output representation 
    :param pos: the position of the output in rank 
    :param features: the (empty) feature extractor utility (where the features go)
    """

    cdef int i,j,k
    cdef str lang = config.lang

    ## background data
    cdef np.ndarray rank_list = config.ranks
    cdef np.ndarray rank_surface = config.rank_vals
    cdef np.ndarray trees = config.tree_pos

    ## foreign representation 
    cdef int[:] output_v = rank_list[output_id]
    cdef unicode foreign = np.unicode(rank_surface[output_id])
    cdef int[:] tree_pos = trees[output_id]
    cdef int ftree,prevtree

    ## input/output lengths and list 
    cdef elen = input_v.shape[0]
    cdef flen = output_v.shape[0]
    cdef list elist = english.split()
    cdef list flist = [u"<unknown>"]+foreign.split()

    ## feature type information
    cdef bint phrase_f    = config.has_phrase
    cdef bint hiero_f     = config.has_hiero
    cdef bint compose_f   = config.has_compose
    cdef bint knowledge_f = config.has_knowledge
    cdef unicode eword,fword,ebigram,fbigram
    cdef int fvlen = config.flen
    cdef long pair_id
    cdef int eid,fid

    ## word pair
    cdef PhrasePair word_component

    ## switches
    cdef bint in_alignment
    cdef bint is_stop
    cdef bint pair_matches
    cdef int class_id
    cdef bint pdbigram

    ## descriptions
    cdef dict description_pairs = config.descriptions
    cdef dict abstract_classes = config.abstract_classes
    cdef int[:] dseq = np.zeros((elen,),dtype=np.int32)
    cdef dict classes = config.class_items
    cdef int num_classes = config.num_classes,cpid,dpid,aid,apid
    cdef dict class_sequence = classes.get(output_id,{})
    cdef int[:] cseq = np.ndarray((flen,),dtype=np.int32)
    cdef int[:] aseq = np.ndarray((flen,),dtype=np.int32)
    cdef bint already_checked
    cseq[0] = -1
    aseq[0] = -1

    cdef str heuristic = config.heuristic
    cdef bint old_model = False if not config.old_model else config.old_model

    ## alignment
    cdef SymAlign alignment
    cdef double[:,:] main_alignment

    ## negative features
    cdef int[:] negs = np.array([4],dtype=np.int32)

    ## global class
    cdef int gclass = -1

    ## rank position features (very important)
    __rank_features(pos,features)

    ## run alignment
    #alignment = base._decode(output_v,input_v,np.insert(input_v,0,0),heuristic=heuristic)
    alignment = base._align(output_v,input_v,np.insert(input_v,0,0),heuristic=heuristic)
    main_alignment = alignment.alignment

    ## input/output length features
    __len_feat(elen,flen,features)
        
    ## relative length of components/english input

    for j in range(1,flen):
        fid = output_v[j]
        class_id = -1
        if j-1 in <dict>class_sequence:
            class_id = class_sequence[j-1]
            gclass = class_id
        cseq[j] = class_id
        ## might want to limit to content words
        __relative_len(elen,fid,class_id,old_model,features)

    ## word pair features

    for i in range(elen):
        eid = input_v[i] ## -1 if oov
        eword = elist[i]

        # e side bigram 
        ebigram = __create_e_bigram(i,elist)
        is_stop = False

        for j in range(1,flen):
            
            ## binary switches
            in_alignment   = False
            pair_matches   = False
            class_id       = -1
            cpid = dpid    = -1
            pdbigram       = False
            prevtree       = -1
            aid            = -1
            apid           = -1

            ## component word information 
            fid    = output_v[j] ## -1 if oov
            fword  = flist[j]
            ftree  = tree_pos[j]
            if j > 1: prevtree = tree_pos[j-1]

            ## pair information 
            pair_id        = word_index(eid,fid,fvlen)
            word_component = ConstructPair(eword,fword,lang=lang)

            ## pair information 
            pair_matches = word_component.sides_match()
            is_stop      = word_component.is_stop_word()
            fbigram      = __create_f_bigram(j,flist)

            ## alignment information (in alignment after heuristic applies)
            if isfinite(main_alignment[i][j]):
                in_alignment = True

            ## in a description?
            if (eword,fword) in description_pairs:
                dseq[i] = 1
                dpid = description_pairs[(eword,fword)]
                if i >= 1 and dseq[i-1] == 1 and description_pairs.get((elist[i-1],fword),False):
                    pdbigram = True

            ## abstract classes?
            if fword in abstract_classes:
                aid = abstract_classes[fword]
                apid = abstract_index(aid,eid,elen)
                aseq[j] = aid
            else:
                aseq[j] = -1
                
            ## name in a class sequence
            if cseq[j] > -1:
                class_id = cseq[j]
                cpid = cat_index(eid,class_id,num_classes)
                
            ## compute word match features
            __word_feat(word_component,pair_matches,
                            eid,fid,pair_id,
                            is_stop,ftree,in_alignment,features)

            # compute bigram features 
            __bigram_features(eword,fword,ebigram,fbigram,ftree,prevtree,pdbigram,fid,features)

            ## compute knowledge features
            if knowledge_f:
                __knowledge_features(dseq[i],dpid,class_id,cpid,aid,apid,eid,fid, # identifiers 
                                         pair_matches,is_stop, ## binary values
                                         ftree,in_alignment, ## sequence information
                                         features)

    if phrase_f:
        phrase_search(config,alignment,elen,flen,hiero_f,elist,flist,
                          dseq,cseq,tree_pos,aseq, ## sequence information
                          gclass,
                          features)

    ## compute binned features
    __compute_bins(features)
    ## number of alignment positions
    
    ## add feature about number of features
    #print features.feature_size

#### PHRASE LEVEL FEATURE COMPUTATION

cdef inline long __ephrase_class(int class_id,int english_id,long num_phrases):
    """Compute the identifier associated with a english phrase and class 

    :param class_id: the identifier of the class 
    :param english_id: the identifier of the english phrase (-1 if unknown)
    :param num_phrases: the number of known phrases
    """
    if english_id == -1 or class_id == -1:
        return -1
    return (class_id*num_phrases)+english_id

cdef inline long __ephrase_asymbol(int aid,int english_id,long num_phrases):
    """Compute a unique idnetifier for an abstract symbol and english phrase pair

    
    :param aid: the abstract symbol id 
    :param english_id: the identifier of the associated english phrase 
    :num_phrases: the number of english phrases
    """
    if english_id == -1 or aid == -1:
        return -1
    return (aid*num_phrases)+english_id


cdef inline void phrase_length_computation(int ilen,int template_id,FeatureMap features):
    """Compute features related to relative length 

    :param ilen: the input length 
    :param template_id: the template id 
    """
    if ilen == 1: features.add_binary(template_id,0)
    elif 2 <= ilen < 4: features.add_binary(template_id,1)
    elif 4 <= ilen < 6: features.add_binary(template_id,2)
    elif 6 <= ilen < 8: features.add_binary(template_id,3)
    elif ilen >= 8: features.add_binary(template_id,4)

cdef inline void __phrase_features(PhrasePair pair,int pid,int epid,int fpid,int aspair, ## idnetifiers
                                       int ftree, ## tree information
                                       int elen,int flen, ## phrase lengths
                                       bint match, bint stop,bint indescr, ## features 
                                       long clpair_id,FeatureMap features):
    """Features associated with raw phrase pairs 

    -- Note that we are discriminating against phrases of length 1, 
    (i.e., english and foreign sides are both equal to 1) since these
    pairs will be handled represented under the word-level features, and 
    phrases that are just stop words on the english side

    :param pair: the phrase pair object 
    :param pairid: the identifier of the phrase pair (-1 if unknown) 
    :param epid: the english phrase id (-1 if unknown)
    :param fpid: the foreign phrase id (-1 if unknown)
    :param aspair: the abstract symbol phrase pair identifier

    :param ftree: the tree position of f side (-1 if not matched)

    :param elen: the length of the english phrase 
    :param flen: the length of the foreign phrases


    :param pid: the phrase id (-1 if unknown)
    :param match: the phrase pair matches 
    :param stop: the english side is a stop (or non content) word
    :param indescr: english side all occurs in descriptions 
 
    :param clpair: the class and phrase pair id (if known)

    :param features: the current features
    """
    cdef bint known = True if pid > -1 else False
    cdef bint clpair = True if clpair_id > -1 else False
    cdef bint great_one = True if (elen > 1 or flen > 1) else False
    cdef bint tree_match = True if ftree != -1 else False
    cdef int word_overlap

    ###########################
    # phrase match and overlap
    ###########################

    ## (do not consider phrases both just words, or stop words on the english side)
    if not great_one or stop:
        return

    word_overlap = pair.word_overlap()
    
    ## match (phrases larger than one)
    if match:
        features.add_binned(17)
        phrase_length_computation(elen,56,features)
        phrase_length_computation(flen,57,features)
        ## tree position
        if tree_match: features.add_binary(69,ftree)

    ## overlap (phrases larger than one)
    elif pair.econtainsf() or pair.fcontainse():
        features.add_binned(18)
        phrase_length_computation(elen,60,features)
        phrase_length_computation(flen,61,features)
        ## tree position
        if tree_match: features.add_binary(71,ftree)
            
    ## overlapping words
    elif word_overlap > 0:
        phrase_length_computation(word_overlap,73,features)
        if tree_match: features.add_binary(72,ftree)
            
    ###########################
    # known phrases 
    ###########################

    ## known phrases greater than 1 
    if known:
        features.add_binary(15,pid)
        features.add_binned(16)
        phrase_length_computation(elen,58,features)
        phrase_length_computation(flen,59,features)

        ## tree position
        if tree_match: features.add_binary(78,ftree)
        
    ## phrase/abstract class pairs 
    if clpair:
        features.add_binary(43,clpair_id)
        
    ## descriptions
    if indescr:
        features.add_binned(67)
        if (flen > 1 and fpid > -1):
            features.add_binary(63,fpid)
        if (elen > 1 and epid > -1):
            features.add_binary(62,epid)
            
        if tree_match: features.add_binary(70,ftree)

    ## abstract symbols and phrase pairs
    if aspair > -1:
        features.add_binary(80,aspair)
        features.add_binned(82)

cdef void __phrase_len_features(int elen,int fpid,int fsidelen,
                                    int ftree1,int ftree2, ## tree information
                                    FeatureMap features):
    """Computes phrases related to the relative length of phrases,sentences,...

    :param elen: the length of the english input 
    :param fpid: the foreign phrase id 
    :param fsidelen: the length of the foregin phrase 
    
    :param features: the current feature map
    """
    cdef int treedis = ftree2-ftree1
    cdef int start_val = (fpid*5)

    if fpid != -1 and fsidelen > 1:
        if 1 <= elen >= 5: features.add_binary(74,start_val)
        if elen >= 10: features.add_binary(74,start_val+1)
        if elen >= 15: features.add_binary(74,start_val+2)
        if elen >= 20: features.add_binary(74,start_val+3)
        if elen >= 25: features.add_binary(74,start_val+4)
        ## the tree gap between phrases 
        features.add_binary(75,treedis)


cdef inline void __hiero_features(unicode lhs, ## lhs side rule 
                              unicode eside,unicode fside,## phrase content
                              dict hiero_rules, ## hiero grammar
                              FeatureMap features):
    """Features related to lexical (i.e., without non-terminal) hierarchical phrase rules 

    -- note these rules are essentially phrase rule, with more information about tree, 
    therefore we do not extract so much information from them. 

    :param lhs: the left hand side of the rule 
    :param eside: the english side of the rule 
    :param fside: the foreign side of the rule 
    :param hiero_rules: the hiero grammar 
    :param features: the current feature set 
    """
    #cdef HieroRule rule = ConstructHPair(lhs,eside,fside)
    cdef tuple t = (lhs.strip(),eside.strip(),fside.strip())
    cdef int identifier,freq

    ## conditions
    #cdef bint sides_match = rule.sides_match()
    
    ## check known hiero rules 
    if t in hiero_rules:
        identifier,freq = hiero_rules[t]
        features.add_binary(22,identifier)

cdef inline void __hiero_gap_rule(unicode lhs,
                                     unicode eside,unicode fside, ## rule sides 
                                     dict hiero_rules, ##hiero rules
                                     bint reorder, ## reordering
                                     int class_id, ## class information
                                     int num_hiero, ## number of hiero rules
                                     dict knowne,dict knownf, # known e and f side rules
                                     dict glue, ## the id of the left hand side of hiero rule
                                     int nume,int numf,
                                     str lang, ## the language of the e side 
                                     FeatureMap features
                                      ):
    """Features associated with more complex hiero rules

    
    :param lhs: left hand side of hiero rule 
    :param eside: the english string in rule (with NTs)
    :param fside: the foreign string in rule (with NTs) 
    :param features: the current feature map 
    """
    cdef HieroRule rule = ConstructHPair(lhs,eside,fside,lang=lang)
    cdef tuple t = (lhs,eside.strip(),fside.strip())
    cdef int identifier,freq,classhid
    cdef int eknown = knowne.get(eside,-1)
    cdef int fknown = knownf.get(fside,-1)
    cdef int lhs_id = glue.get(lhs)

    ## is english side is just a stop word 
    cdef bint is_stop = rule.is_stop_word()
    ## if rule is only terminal rules, this is not very informative
    cdef bint only_terminals = rule.only_terminals()
    cdef bint eterm = rule.left_terminal_only()
    cdef bint fterm = rule.right_terminal_only()
    
    ## known rules 
    if t in hiero_rules and not only_terminals and not is_stop:
        identifier,freq = hiero_rules[t]
        features.add_binary(22,identifier)
        features.add_binned(23)

        ## hiero rules related to classes 
        if class_id != -1:
            classhid = (class_id*num_hiero)+identifier
            features.add_binary(45,classhid)
            
    # ## unknown rules (still can extract general properties)
    else:
        features.add_binned(92)

    # # ## reordering 
    if reorder:
        features.add_binned(29)
        features.add_binary(30,lhs_id)
        if class_id != -1: features.add_binned(91)
            
    # # ## english side known?
    if eknown != -1 and not eterm and not is_stop:
        features.add_binary(89,eknown)
        if class_id != -1: features.add_binary(93,(class_id*nume)+eknown)

    # # ## foreign side known?
    if fknown != -1 and not fterm:
        features.add_binary(90,fknown)
        if class_id != -1: features.add_binary(94,(class_id*numf)+fknown)
                
cdef int phrase_search(object config,SymAlign alignment,int elen,int flen,bint hiero,
                            list einput,list finput, ## raw inputs
                            int[:] descriptions,int[:] classes,int[:] treepos,int[:] abstracts, ## sequences
                            int gclass, ## global class id 
                            FeatureMap features) except -1:
    """Find phrases for a given input and alignment and (optionally) find hiero rules


    -- enumerates the phrases consistent with alignment, checks if they are known phrases,
    and extract a number of related features (e.g., match features, description features,...)

    :param config: the main configuration file 
    :param alignment: the symmetric alignment
    :param elen: the length of the english input 
    :param flen: the length of the foreign input 
    :param einput: the english input 
    :param finput: the foreign input 

    :param hiero: bool indiciating whether to do the hierarchical rule search 
    :param descriptions: positions of descriptions 
    :param classes: positions of abstract see-also classes
    :param treepos: the tree positions of component words
 
    :param features: the current feature map
    """
    cdef Phrases phrases_found = alignment.extract_phrases(7)
    cdef int[:,:] phrase_loc = phrases_found.phrases
    cdef int num_phrases = phrase_loc.shape[0]
    cdef int i,j,k
    cdef unicode english_side,foreign_side
    cdef str lang = config.lang

    ## example phrases 
    cdef PhrasePair phrase_instance

    ## known phrases
    cdef dict known_phrases = config.phrase_map
    cdef int phrase_id,english_pid,foreign_pid,estart,eend,fstart,fend
    cdef dict english_phrases = config.english_map
    cdef dict foreign_phrases = config.foreign_map
    cdef int num_en_phrases = config.num_en_phrases
    cdef int num_fr_phrases = config.num_fr_phrases
    cdef long clpair

    ## phrase properties
    cdef bint phrase_matches
    cdef int fsidelen,esidelen
    cdef int ftree
    cdef int class_id,aspair
    cdef int sfindescr

    ### see also classes
    cdef int num_classes = config.num_classes
        
    ## phrase charts 
    cdef list echart #,fchart

    ## glue grammar 
    cdef dict rule_map = config.glue
    cdef dict hiero_rules = config.hiero
    cdef dict lookup_table = {}

    ## hiero stuff
    cdef tuple tuple1
    cdef unicode tlh1,lhs
    cdef int edist,fdist
    cdef int w
    
    ## with alignment
    cdef int[:] e_aligned,f_aligned

    e_aligned = np.zeros((elen,),dtype=np.int32)
    f_aligned = np.zeros((flen,),dtype=np.int32)
    echart = [[u'' for _ in range(elen+1)] for _ in range(elen)]

    ## normal phrase search
    
    for i in range(num_phrases):

        #conditions
        ftree     = -1
        class_id  = -1
        clpair    = -1
        sfindescr = False
        aspair    = -1
        
        ## phrase positions in different strings 
        estart = phrase_loc[i][0]
        eend   = phrase_loc[i][1]
        fstart = phrase_loc[i][2]
        fend   = phrase_loc[i][3]

        ## check alignment points
        for w in range(estart,eend+1): e_aligned[w] = 1
        for w in range(fstart,fend+1): f_aligned[w] = 1

        ## the english and foreign sides
        english_side = to_unicode(' '.join([einput[k] for k in range(estart,eend+1)]).strip())
        foreign_side = to_unicode(' '.join([finput[k] for k in range(fstart,fend+1)]).strip())
        
        ## length of foreign phrase 
        fsidelen = fend+1 - fstart
        esidelen = eend+1 - estart

        ## the phrase pair 
        phrase_instance = ConstructPair(english_side,foreign_side,lang=lang)

        ## known phrase or not...
        phrase_id = known_phrases.get((english_side,foreign_side),-1)
        phrase_matches = True if phrase_instance.sides_match() else False
        is_stop = True if phrase_instance.is_stop_word() else False

        ## phrase identifiers 
        english_pid = english_phrases.get(english_side,-1)
        foreign_pid = foreign_phrases.get(foreign_side,-1)

        ## phrase consistent with tree? (should be larger than 1)
        if treepos[fstart] == treepos[fend]:
            ftree = treepos[fstart]

        ## foreign side is consistent with an abstract class?
        if classes[fstart] > -1 and classes[fstart] == classes[fend]:
            class_id = classes[fstart]
            clpair   = __ephrase_class(class_id,english_pid,num_en_phrases)

        ## foreign side is consistent with an abstract symbol type?
        if abstracts[fstart] > -1 and abstracts[fstart] == abstracts[fend]:
            aspair = __ephrase_asymbol(abstracts[fstart],english_pid,num_en_phrases)
            
        ## english start and finish are in descriptions (should be larger than 1)
        if descriptions[estart] == 1 and descriptions[estart] == descriptions[eend]:
            sfindescr = True

        ## phrase sentence length
        #__phrase_len_features(elen,foreign_pid,fsidelen,treepos[fstart],treepos[fend],features)

        ## phrase features computation (match,overlap,...)
        
        __phrase_features(phrase_instance,phrase_id,english_pid,foreign_pid,aspair, ## identifiers
                              ftree, ## tree information
                              esidelen,fsidelen, ## length information
                              phrase_matches,is_stop,sfindescr,
                              clpair,features)

        ## hiero phrase for chart 
        if hiero:

            ## tree position (typle)
            tuple1 = (str(treepos[fstart]),str(treepos[fend]))
            tlhs1  = rule_map.get(tuple([str(treepos[fstart])]),u'')
            edist = eend - estart
            fdist = fend - fstart

            ## smallest span 
            if estart == eend:
                lhs = tlhs1

                ## single tree span 
                if ftree != -1 and lhs:
                    echart[estart][eend+1] = lhs
                    lookup_table[(estart,eend+1)] = (fstart,fend+1)

                    ## rule is already a phrase rule, don't need to add
                    
                    ## add hiero phrase feature
                    #__hiero_features(lhs,english_side,foreign_side,hiero_rules,features)
                                                            
                ## multiple tree spans 
                elif tuple1 in rule_map and fdist <= 3:
                    lhs = rule_map[tuple1]
                    
                    ## left 
                    if fstart >= 1 and treepos[fstart-1] == treepos[fstart]: fstart += 1
                    ## right
                    if treepos[fend-2] == treepos[fend-1]: fend -= 1
                    ## new text span
                    foreign_side = to_unicode(' '.join([finput[k] for k in range(fstart,fend+1)]).strip())
                    if not foreign_side: continue                     
                    lookup_table[(estart,eend+1)] = (fstart,fend+1)
                    echart[estart][eend+1] = lhs

                    ## add phrase feature
                    #__hiero_features(lhs,english_side,foreign_side,hiero_rules,features)
                    
                ## can be fixed with one shift to the right
                elif not fend-1 <= fstart and treepos[fend-1] == ftree and lhs:
                    lookup_table[(estart,eend+1)] = (fstart,fend)
                    echart[estart][eend+1] = lhs

                    ## a shift here, might not be captured in phrase rules
                    
                    ## add phrase feature
                    #__hiero_features(lhs,english_side,foreign_side,hiero_rules,features)

            ## long spans with abstract lhs
            elif edist <= 3 and ftree != -1 and tlhs1:
                echart[estart][eend+1] = tlhs1
                lookup_table[(estart,eend+1)] = (fstart,fend+1)

                ## add phrase feature
                #__hiero_features(tlhs1,english_side,foreign_side,hiero_rules,features)

            ## longer spans with rhs
            elif edist <= 3 and tuple1 in rule_map and fdist <= 3:
                lhs = rule_map[tuple1]

                ## check that the tree is not just a fragment and shifts if needed
                
                # left 
                if fstart >= 1 and treepos[fstart-1] == treepos[fstart]: fstart += 1
                # right
                if treepos[fend-2] == treepos[fend-1]: fend -= 1

                ## new foreign side 
                foreign_side = to_unicode(' '.join([finput[k] for k in range(fstart,fend+1)]).strip())
                if not foreign_side: continue

                ## should be a pair of spans 
                #lookup_table[(estart,eend+1)] = lhs
                lookup_table[(estart,eend+1)] = (fstart,fend+1)
                
                echart[estart][eend+1] = lhs
                ## add phrase feature
                #__hiero_features(lhs,english_side,foreign_side,hiero_rules,features)

    #hiero search (if specified)
    if hiero:
        hiero_search(config, ## configuration 
                         elen,flen,
                         finput,einput,
                         echart, ## current chart
                         lookup_table,#rule_map,#hiero_rules, ## hiero infromation
                         e_aligned,f_aligned,## alignment points 
                         gclass,treepos, ## class and tree information
                         features)

## move left

cdef inline int move_left(int start,int barrier,int[:] aligned):
    """Shift a span left to include unaligned items

    :param start: the starting point 
    :param barier: the point to end at
    :param aligned: the points that are aligned
    """
    cdef int nstart = start

    while True:
        if nstart == barrier:
            break
        if aligned[nstart-1] == 1 or (start - nstart) > 3:
            break
        nstart -= 1

    return nstart

cdef inline int move_right(int end,int elen,int[:] aligned):
    """Shift a span left to include unaligned items

    :param end: the end of the sequence 
    :param elen: the length of the input 
    :param aligned: the alignment points
    """
    cdef int nend = end

    while True:
        if nend >= elen:
            break
        if aligned[nend] == 1 or (nend - end ) > 3:
            break
        nend += 1
    return nend


cdef inline int find_start(int start,int end,int[:] aligned):
    """Find a new starting value for an unknown span 

    :param span: the starting span 
    """
    cdef int nstart = start
    
    while True:
        if nstart+1 >= end or (nstart-start) >= 3:
            nstart = start
            break
        if aligned[nstart] == 1:
            break
        nstart += 1
    return nstart

cdef inline int find_end(int start,int end,int[:] aligned):
    """Find a new end point"""
    cdef int nend = end

    while True:
        if nend - 1 <= start or (end-nend) >= 3:
            nend = end
            break
        if aligned[nend-1] == 1:
            break
        nend -= 1
    return nend

cdef inline unicode make_gap(list tinput,
                                 int start,int nstart, ## start points
                                 int end,int nend,
                                 unicode span1,unicode span2):
    """Create a representaiton with gaps 

    :param tinput: the text input 
    :param start: the starting pont 
    :param nstart: the new starting point 
    :param end: the original end point 
    :param nend: the new end point 
    :param span1: the tag for span1
    :param span2: the tag for span 2
    """
    cdef int w

    return to_unicode(' '.join([tinput[w] for w in range(nstart,start)]+\
        ["[%s_1]" % span1,"[%s_2]" % span2]+[tinput[w] for w in range(end,nend)]))
      
    
cdef inline unicode make_fgap(list tinput,
                                 int start,int nstart,
                                 int g1,int g2, ## middle gap
                                 int end,int nend,
                                 unicode span1, unicode span2
                                 ):
    """Create a representaiton with gaps for the foreign sentence  

    :param tinput: the text input 
    :param start: the starting pont 
    :param nstart: the new starting point 
    :param g1: middle gap start point
    :param g2: midd gap end point 
    :param end: the original end point 
    :param nend: the new end point 
    :param span1: the tag for span1
    :param span2: the tag for span 2
    """
    cdef int w

    return to_unicode(' '.join([tinput[w] for w in range(nstart,start)]+[span1]+\
        [tinput[w] for w in range(g1,g2)]+[span2]+\
        [tinput[w] for w in range(end,nend)]))

##

cdef inline int ftree_move_left(int fstart,int ftree,int[:] ftreepos):
    """Move the f start position left"""
    cdef int nfstart = fstart

    while True:
        if nfstart <= 1 or ftreepos[nfstart-1] != ftree or (fstart-nfstart) >= 3:
            break
        nfstart -= 1

    return nfstart

cdef inline int ftree_move_right(int end,int flen,int ftree,int[:] ftreepos):
    """Move the f start position left"""
    cdef int nfend = end

    while True:
        if nfend+1 >= flen or ftreepos[nfend+1] != ftree or (nfend-end) >= 3:
            break
        nfend += 1 
    return nfend


## hiero search function

cdef inline int hiero_search(object config, ## configuration, hold a lot of information 
                                  int elen,int flen,
                                  list finput,list einput, ## the actual input 
                                  list echart,
                                  dict lookup_table,
                                  int[:] e_aligned,int[:] f_aligned, ## alignment points
                                  int gclass,int[:] trees,
                                  FeatureMap features ## current features
                            ) except -1:
    """A simplified chart search procedure for finding hiero rules
    
    :param elen: the length of the english input 
    :param echart: the english chart
    :param lookup_table: points to foreign projections of english rules
    :param rule_map: the glue grammar 

    :param features: the current feature set
    """
    cdef int spansize,start,end,mid
    cdef unicode span1,span2,lhs
    cdef int fs1,fs2,fe1,fe2
    cdef int nstart,nend,g1,g2
    cdef int nfstart,nfend
    cdef unicode espan,fspan,tag
    cdef int w
    cdef bint reordering
    cdef int fstart,fend,ftree
    
    ## hiero related information 
    cdef int num_hiero = config.num_hiero
    cdef dict hiero_rules = config.hiero
    cdef dict rule_map = config.glue
    cdef dict lhs_glue = config.lhs_glue
    
    ## hiero sides 
    cdef dict hes = config.hes
    cdef dict hfs = config.hfs
    cdef int numhe = config.numhe
    cdef int numhf = config.numhf
    ## lang
    cdef str lang = config.lang
    
    for spansize in range(2,elen+1):
        for start in range(elen - spansize+1):
            end = start + spansize

            ## unknown spans, add with contexts 
            if (start,end) not in lookup_table:
                nstart = start
                nend = end

                if e_aligned[start] != 1 or e_aligned[end-1] != 1:
                    ##find start
                    nstart = find_start(start,end,e_aligned)
                    ## find end
                    nend = find_end(start,end,e_aligned)

                ## do this form a known span?
                tag = echart[nstart][nend]

                ## tag found with new positions
                if tag:
                    fstart,fend = lookup_table[(nstart,nend)]
                    ftree = trees[fstart]

                    nfstart = fstart
                    nfend = fend
                    
                    ## move right
                    nfstart = ftree_move_left(fstart,trees[fstart],trees)
                    ## move left
                    nfend = ftree_move_right(fend,flen,trees[fstart],trees)

                    espan = to_unicode(' '.join([einput[w] for w in range(start,nstart)]+["[%s]" % tag]+\
                      [einput[w] for w in range(nend,end)]))
                    fspan = to_unicode(' '.join([finput[w] for w in range(nfstart,fstart)]+["[%s]" % tag]+\
                      [finput[w] for w in range(fend,nfend) if w < flen]))
                      
                    echart[start][end] = tag
                    lookup_table[(start,end)] = (nfstart,nfend)

                    ## extract rule features
                    __hiero_gap_rule(tag,espan,fspan,hiero_rules,False,gclass,num_hiero,hes,hfs,lhs_glue,
                                         numhe,numhf,lang,features)
                                        
            ## middle point 
            for mid in range(start+1,end):
                span1 = echart[start][mid]
                span2 = echart[mid][end]

                ## no patterns found 
                if (not span1 or not span2) or (span1,span2) not in rule_map:
                    continue

                ## found left hand side rule 
                lhs = rule_map[(span1,span2)]

                fs1,fe1 = lookup_table[(start,mid)]
                fs2,fe2 = lookup_table[(mid,end)]
                if (fs1,fe1) == (fs2,fe2): continue
                fstart = fs1 if fs1 < fs2 else fs2
                fend = fe1 if fs1 > fs2 else fe2

                ## see if there is a middle gap
                
                if (fs1 == fstart):
                    if fs2 < fe1: continue
                    g1 = fe1; g2 = fs2
                elif (fs2 == fstart):
                    if fs1 < fe2: continue
                    g1 = fe2; g2 = fs1
                if g2 - g1 > 3: continue

                reordering = False
                    
                ### move e 
                nstart = move_left(start,0,e_aligned)
                nend   = move_right(end,elen,e_aligned)
                
                # move f 
                nfstart = move_left(fstart,1,f_aligned)
                nfend = move_right(fend,flen,f_aligned)

                ## add to chart, update lookup table 
                echart[nstart][nend] = lhs
                lookup_table[(nstart,nend)] = (nfstart,nfend)

                # if (nend - nstart) <= 3 and (nfend - nfstart) <= 3:
                #     espan = ' '.join([einput[w] for w in range(nstart,nend)])
                #     fspan = ' '.join([finput[w] for w in range(nfstart,nfend)])
                #     #__hiero_features(lhs,espan,fspan,hiero_rules,features)

                ## generic span
                ## make a espan with gap
                espan = make_gap(einput,start,nstart,end,nend,span1,span2)

                ## figure out ordreing on fside
                
                if fstart == fs1:
                    fspan = make_fgap(finput,fstart,nfstart,g1,g2,fend,nfend,
                                          u"[%s_1]" % span1,u"[%s_2]" % span2)

                else:
                    fspan = make_fgap(finput,fstart,nfstart,g1,g2,fend,nfend,
                                          u"[%s_2]" % span2,u"[%s_1]" % span1)
                    reordering = True
                    
                ## main gap rule
                __hiero_gap_rule(lhs,espan,fspan,hiero_rules,reordering,gclass,num_hiero,hes,hfs,lhs_glue,
                                     numhe,numhf,lang,features)

                if (mid - start) <= 2:
                    espan = to_unicode(' '.join([einput[w] for w in range(start,mid)]+["[%s_2]" % span2]))
                    __hiero_gap_rule(lhs,espan,fspan,hiero_rules,reordering,gclass,num_hiero,hes,hfs,lhs_glue,
                                         numhe,numhf,lang,features)

                    if reordering:
                        __hiero_gap_rule(lhs,espan,u"[%s_1] [%s_2]" % (span1,span2),
                                             hiero_rules,reordering,gclass,num_hiero,hes,hfs,lhs_glue,
                                             numhe,numhf,lang,features)
                    else:
                        __hiero_gap_rule(lhs,espan,u"[%s_2] [%s_1]" % (span2,span1),
                                             hiero_rules,reordering,gclass,num_hiero,hes,hfs,lhs_glue,
                                             numhe,numhf,lang,features)
                    
                if (end - mid) <= 2:
                    espan = to_unicode(' '.join([u"[%s_1]" % span1]+[einput[w] for w in range(mid,end)]))
                    __hiero_gap_rule(lhs,espan,fspan,hiero_rules,reordering,gclass,num_hiero,hes,hfs,lhs_glue,
                                         numhe,numhf,lang,features)
                    
                    if reordering:
                        __hiero_gap_rule(lhs,espan,u"[%s_1] [%s_2]" % (span1,span2),
                                             hiero_rules,reordering,gclass,num_hiero,hes,hfs,lhs_glue,
                                             numhe,numhf,lang,features)
                    else:
                        __hiero_gap_rule(lhs,espan,u"[%s_2] [%s_1]" % (span2,span1),
                                             hiero_rules,reordering,gclass,num_hiero,hes,hfs,lhs_glue,
                                             numhe,numhf,lang,features)

### CLI related items


def params():
    """Defines the different parameters for the aligner extractor

    :rtype: tuple 
    :returns: description of switches and default settings
    """
    options = [
        ("--atemplates","atemplates","lex+phrase+align+knowledge+hiero+compose","str",
             "types of aligner templates [default=lex+phrase+hiero+knowledge+compose]","AlignerExtractor"),
        ("--hierogrammar","hierogrammar",'',"str",
             "location of hiero grammar [default='']","AlignerExtractor"),
         ("--gluegrammar","gluegrammar",'',"str",
              "location of glue grammar [default='']","AlignerExtractor"),
         ## have only one beam
         ("--beam","beam",200,"int",
              "beam size for training and testing reranker [default=200]","AlignerExtractor"),
         ("--lang","lang",'en',"str",
              "the language of the reranker [default='en']","AlignerExtractor"),
         ("--store_feat","store_feat",False,"bool",
              "extract features only once to file  [default=False]","AlignerExtractor"),
        ("--temp_blocks","temp_blocks","","str",
             "Feature templates to block by id [default='']","AlignerExtractor"),
        ("--feat_blocks","feat_blocks","","str",
             "Individual features to block by id [default='']","AlignerExtractor"),
        ("--old_model","old_model",False,"bool",
             "Revert to old model (older feature set) [default=False]","AlignerExtractor"),
    ]

    extractor_group = {"AlignerExtractor":"Feature Extractor Settings"}
    return (extractor_group,options)
