# -*- coding: utf-8 -*-

"""

This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Feature extractor for graph decoders 

There is a lot of redundancy with the AlignExtract code and routines, which was
implemented with less generalization in mind. The aim is to replace that code with
this code.  

"""

import os
import sys
import gzip
import time
import pickle
import re
import numpy as np
cimport numpy as np
from zubr.Extractor cimport Extractor
from zubr.Features cimport FeatureObj,FeatureMap,TemplateManager
from zubr.util.config import ConfigAttrs
from cython cimport boundscheck,wraparound,cdivision
from zubr.GraphDecoder cimport WordGraphDecoder
from zubr.GraphDecoder import Decoder
from zubr.util.graph_extractor import *
from zubr.util.config import ConfigAttrs
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Dataset cimport RankPair,RankStorage,RankComparison,RankDataset
from zubr.SymmetricAlignment cimport SymmetricWordModel,SymAlign,Phrases
from zubr.util.config import ConfigAttrs,ConfigObj

from zubr.Phrases cimport (
    SparseDictWordPairs,
    DescriptionWordPairs,
    SimplePhraseTable,
    HieroPhraseTable,
    PhrasePair,
    ConstructPair,
    ConstructHPair,
    HieroRule,
)

from zubr.FeatureComponents cimport (
    WordPhraseComponents,
    KnowledgeComponents,
    RankComponents,
    StorageComponents,
    MonoComponents,
    PolyComponents,
)

## classes

cdef class GraphExtractorBase(Extractor):
    """Base class for graph feature extractors"""

    cdef FeatureObj extract_from_scratch(self,RankPair instance,str etype):
        """Extract features from scratch 

        :param instance: an data instance to extract an item for
        :param etype: the type of data to extract for 
        """
        raise NotImplementedError

    cdef FeatureObj extract_from_file(self,str directory, str etype,int identifier):
        """Extract features backed up to a file 

        :param directory: the directory where the features sit 
        :param etype: the type of data being used (e.g., train/test/valid...)
        :param identifier: 
        """
        raise NotImplementedError

    def backup(self,wdir):
        """Back up the extractor using a custom protocol 

        :param wdir: the working directory, place to put backup files 
        """
        raise NotImplementedError

    def load_backup(self,config):
        """Load items from a backup 

        :param config: the main experiment configuration 
        """
        raise NotImplementedError

    cpdef void offline_init(self,object dataset,str rtype):
        """This function either extracts features using a number of 
        subprocesses (when the extraction takes a long time) or passes. 

        Idea: Feature extraction is sometimes prohibitively slow, especially 
        for large datasets. Using an idea similar to the ConcurDecoder implementation
        in Zubr.GraphDecoder, this function has the option of doing offline feature 
        extraction by splitting the dataset into smaller parts, and extracting features
        for each part asynchronously using low-level, unix system calls back to the 
        Zubr FeatureExtractor. 
        
        If the switch ``extract_concur`` is not set, this will simply be passed
        """
        if self._config.concur_extract:
            if rtype == 'valid' and not self._config.val_extract:
                run_concurrent(self._config,dataset,rtype)
                self._config.val_extract = True
            elif rtype == 'train':
                run_concurrent(self._config,dataset,rtype)

    cdef void after_eval(self,RankDataset dataset,RankComparison ranks):
        """Called after the evaluation, computes scores in the case that the dataset 
        contains multiple internal datasets. 

        :param dataset: the dataset just evaluated 
        :param ranks: the reranker output 
        """
        if dataset.mulilingual:
            return
        return
        
    def exit(self):
        if hasattr(self.decoder,'exit'):
            self.decoder.exit()

    property dir:
        """Information about the directory where the extractor sits"""
        def __get__(self):
            return self._config.dir
        def __set__(self,new_dir):
            self._config.dir = new_dir

    property offset:
        """Information about the directory where the extractor sits"""
        def __get__(self):
            return self._config.offset
        def __set__(self,new_offset):
            self._config.offset = new_offset

cdef class GraphRankExtractor(GraphExtractorBase):
    """A standard graph extractor for rank decoding problems (i.e., where there is some rank list)"""

    def __init__(self,decoder,word_phrase,ranks,knowledge,storage,config):
        """Create a graph rank extractor instance
        
        :param word_phrase: word/phrase components 
        :param ranks: rank components 
        :param knowledge: knowledge componenents 
        :param storage: the container that contains rank storage information 
        :param config: the local extractor configuration
        """
        self.decoder     = decoder
        self.word_phrase = word_phrase
        self.ranks       = ranks
        self.knowledge   = knowledge
        self.storage     = storage 
        self._config     = config

    @classmethod
    def build_extractor(cls,config):
        """Build an extractor from configuration 

        :param config: the main configuration being used 
        """
        ## load the graph decoder 
        dtype = Decoder(config.decoder_type)
        decoder = dtype.load_backup(config)

        ## ## initialize extractor
        extractor_settings = ConfigAttrs()        
        wc,ranks,kc,sc = build_extractor(config,extractor_settings,decoder)

        return cls(decoder,wc,ranks,kc,sc,extractor_settings)

    ## extraction functions

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
        cdef int offset = 0 if not config.offset else config.offset
        
        if from_file and off:
            return self.extract_from_file(config.dir,etype,identifier+offset)
        return self.extract_from_scratch(instance,etype)

    cdef FeatureObj extract_from_file(self,str directory,str etype, int identifier):
        """Extract features representations from a file
        
        -- Each line in the feature representation is a different entry

        :param directory: the target directory 
        :param etype: the type of feature representation (e.g., train,test,...)
        :param identifier: the example identifier
        """
        cdef object config = self._config
        cdef StorageComponents storage = self.storage
        cdef RankStorage ranks = storage.trainranks
        cdef int baseline = -1

        if etype == "train":
            baseline = ranks.gold_position(identifier)
        
        return FeatureObj.load_from_file(config.dir,etype,identifier,
                                             config.num_features,
                                             baseline=baseline,
                                             )

    cdef RankComparison rank_init(self,str etype):
        """Returns a rank comparison instance for evaluation

        :param etype: the type of data (e.g., train/test/valid/...)
        :raises ValueError: when the etype if unknown
        """
        cdef object config = self._config
        cdef int beam = config.beam
        cdef StorageComponents storage = self.storage

        if etype == 'train':
            return RankComparison(storage.trainranks,beam)

        elif etype == 'train-test':
            return RankComparison(storage.trainranks,beam)
        
        elif etype == 'test':
            return RankComparison(storage.testranks,beam)

        elif etype == 'valid' or etype == 'valid-select':
            return RankComparison(storage.validranks,beam)

        elif etype == 'query':
            return RankComparison(storage.queryranks,beam)

        raise ValueError('Etype unknown: %s' % etype)
    
    cdef FeatureObj extract_from_scratch(self,RankPair instance,str etype):
        """Extract features from scratch

        :param instance: training instance 
        :param etype: the dataset type
        """
        cdef object config = self._config
        cdef int beam = config.beam

        ## extractor components
        cdef WordPhraseComponents word_phrase = self.word_phrase
        cdef KnowledgeComponents knowledge = self.knowledge
        cdef WordGraphDecoder decoder = <WordGraphDecoder>self.decoder
                
        #ranks items
        cdef StorageComponents storage = self.storage
        cdef RankStorage ranks
        cdef RankComponents rank_info = <RankComponents>self.ranks

        # data point information
        cdef int identifier = instance.global_id
        cdef unicode surface = instance.surface
        cdef int[:] input_v = instance.en
        cdef int gold_id = instance.rep_id
        
        ## feature templates 
        cdef dict temps = config.tempmanager
        cdef long num_features = config.num_features
        cdef int value_pos
        cdef int offset = 0 if not config.offset else config.offset
        cdef int actual_position
        cdef bint extracted_gold

        ## store features
        cdef bint from_file = config.store_feat

        ## gold information 
        cdef int gold_pos = -1, gold_item = -1

        ## add the offset
        identifier = identifier + offset 

        ## find the correct ranks 
        if etype == 'train':
            ranks = storage.trainranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]
            gold_pos = ranks.gold_position(identifier)

        elif etype == 'valid':
            ranks = storage.validranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]

        elif etype == 'test':
            ranks = storage.testranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]
            
        elif etype == 'query':
            ranks = storage.queryranks
            rank_items = ranks.find_ranks(identifier,etype,beam)
            rank_size = rank_items.shape[0]
            
        else:
            raise ValueError('Unknown etype: %s' % etype)
        
        ## the main features
        features = FeatureObj(rank_size,templates=temps,maximum=num_features)

        #############################
        ### GOLD EXTRACTION (for training)
        
        
        if gold_pos > -1:
            gold_item = ranks.gold_value(identifier)
            try: 
                main_extractor(decoder,
                                   config,
                                   input_v,
                                   surface,
                                   gold_item,
                                   gold_pos,
                                   word_phrase,
                                   knowledge,
                                   rank_info,
                                   <FeatureMap>features.gold_features)

                ## make what the baseline thinks
                features.baseline = gold_pos
                
            except Exception,e:
                self.logger.error(e,exc_info=True)
                sys.exit('Feature Extraction error encountered on gold, check log')

        #############################
        ### the rest

        for value_pos in range(rank_size):
            new_identifier = rank_items[value_pos]
            if gold_pos > -1 and value_pos == gold_pos:
                actual_position = value_pos + 1
            else:
                actual_position = value_pos

            if gold_pos > -1 and new_identifier == gold_item:
                self.logger.warning('Beam contains gold item! skipping...')
                continue 
                #sys.exit('Beam contains gold item, check log')
            try:
                main_extractor(decoder,
                                   config,
                                   input_v,
                                   surface,
                                   new_identifier,
                                   #value_pos,
                                   actual_position,
                                   word_phrase,
                                   knowledge,
                                   rank_info,
                                   <FeatureMap>features[value_pos])
            except Exception,e:
                self.logger.info(e,exc_info=True)
                sys.exit('Feature Extraction error encountered (on non-gold), check log')

        ## backup feature representations to file 
        if from_file and etype != 'query':
            features.print_features(config.dir,
                                        etype,
                                        identifier,
                                        rank_items,
                                        gold_item)

        return features
            
    ## properties

    property num_features:
        """Returns the number of features using extractor"""
        def __get__(self):
            return self._config.num_features

    property config:
        """Stores the ranks for the validation data (if provided)"""

        def __get__(self):
            return self._config
        
    ## backup protocol 

    def backup(self,wdir):
        """Back up the extractor using a custom protocol 

        :param wdir: the working directory, place to put backup files 
        :rtype: None 
        """
        self.logger.info('Backing up the graph extractor...')
        
        # backup the decoder 
        self.decoder.backup(wdir)

        ## create extractor directory
        fdir = os.path.join(wdir,"graph_extractor")
        if os.path.isdir(fdir):
            self.logger.info('Already backed up, skipping...')
            return

        ## make directory 
        os.mkdir(fdir)

        self.logger.info('Backing up individual extractor components')
        ## backup the different components 
        self.word_phrase.backup(fdir)
        self.ranks.backup(fdir)
        self.knowledge.backup(fdir)
        self.storage.backup(fdir)

        ## backup the configuration (this doesn't work for some reason)
        try: 
            self.logger.info('Backing up the extractor configuration object...')
            econfig = os.path.join(fdir,"extractor_config.p")
            with open(econfig,'wb') as my_config:
                pickle.dump(self._config,my_config)
        except Exception,e:
            self.logger.error(e,exc_info=True)

        self.logger.info('Finished backing up...')

    @classmethod 
    def load_backup(cls,config):
        """Load items from a backup 

        :param config: the main experiment configuration 
        :returns: a graph extractor instance 
        """
        ## load the decoder 
        dclass = Decoder(config.decoder_type)
        decoder = dclass.load_backup(config)

        ## extractor directory 
        fdir = os.path.join(config.dir,'graph_extractor')
        wdir = config.dir
        rank_type = PolyComponents if config.ispolyglot else MonoComponents

        config.dir = fdir
        words     = WordPhraseComponents.load_backup(config)
        ranks     = rank_type.load_backup(config)
        knowledge = KnowledgeComponents.load_backup(config)
        storage   = StorageComponents.load_backup(config)

        ## switch back 1
        config.dir = wdir

        ## configuration
        with open(os.path.join(fdir,"extractor_config.p"),'rb') as config:
            settings = pickle.load(config)
        instance =  cls(decoder,words,ranks,knowledge,storage,settings)
        instance.logger.info('Loaded final backup...')
        return instance

## C EXTRACTION METHODS

cdef unicode to_unicode(s):
    if isinstance(s,bytes):
        return (<bytes>s).decode('utf-8')
    return s

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

cdef inline void __lang_rank_features(int langid,int pos,FeatureMap features,int offset):
    """Assign features related to position rank


    :param pos: the position of the item in rank 
    :param features: the current features
    :rtype: None
    """
    if pos == 0:
        features.add_binary(59,offset+0)
        ## add the second as feature to lift up second position stuff
        features.add_binary(59,offset+1)
    elif pos == 1: features.add_binary(59,offset+1)
    elif pos == 2: features.add_binary(59,offset+2)
    elif pos == 3: features.add_binary(59,offset+3)
    elif pos == 4: features.add_binary(59,offset+4)
    elif pos == 5: features.add_binary(59,offset+5)
    elif pos == 6: features.add_binary(59,offset+6)
    elif pos == 7: features.add_binary(59,offset+7)
    elif pos == 8: features.add_binary(59,offset+8)
    elif pos == 9: features.add_binary(59,offset+9)        
    ## rank ranges
    if pos < 5:  features.add_binary(59,offset+10)
    if pos < 10: features.add_binary(59,offset+11)
    if 10 <= pos < 20: features.add_binary(59,offset+12)
    if 20 <= pos < 30: features.add_binary(59,offset+13)
    if pos >= 30: features.add_binary(59,offset+14)
    ## 0 or not
    ## id features
    features.add_binary(71,langid)
    

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
    return to_unicode(''.join(word_list[index-1:index+1])) ## removes the spacing

cdef inline long cat_index(int eid,int aid,int alen):
    """find the feature number for word id

    :param eid: the english identifier 
    :param aid: the abstract class identifier 
    :param alen: the number of abstract classes
    """
    if eid == -1 or aid == -1 or alen == 0: return -1
    return (eid*alen)+aid


cdef inline void __word_feat(PhrasePair pair,
                                 bint match,
                                 int eid,
                                 int fid,
                                 long pair_id, ## identifiers
                                 bint stop,
                                 int treepos,
                                 bint in_alignment, ## properties of pair 
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
        #features.add_binary(2,pair_id)
        features.add_increm_binary(2,pair_id)

    ## word identies
    if not oov and not stop:
        #features.add_binary(10,eid)
        #features.add_binary(11,fid)
        features.add_increm_binary(10,eid)
        features.add_increm_binary(11,fid)
    
    ## viterbi alignment
    if in_alignment and not stop:
        features.add_binary(15,treepos)
        if not oov: features.add_binary(14,pair_id)
        if match: features.add_binned(16)
            
    # ## matching content words 
    # #if match and not stop:
    if match:
        features.add_binned(3)
        features.add_binary(5,treepos)
        if not oov: features.add_binary(4,fid)

    # ## overlap
    elif pair.econtainsf() or pair.fcontainse() and not stop:
        features.add_binned(1)
        #features.add_binary(17,treepos)
        features.add_increm_binary(17,treepos)

# def inline bigram_size()
         

cdef inline void __bigram_features(unicode english,
                                      unicode foreign,
                                      unicode ebigram,
                                      unicode fbigram,
                                      int treepos,
                                      int prevtree,
                                      bint prevdescr,
                                      int fid,
                                      FeatureMap features):
    """Computes various bigram features, such as overlap, match, and adds to feature map


    :param english: the english word
    :param foreign: the foreign word 
    :param ebigram: the english bigram 
    :param fbigram: the foreign bigram 
    ...
    """
    cdef PhrasePair bigram_pair,eword_fbigram,ebigram_fword
    cdef int tree = treepos if treepos == prevtree else -1 

    ## avoid items that match 
    if ebigram and fbigram and english != foreign:
        bigram_pair = ConstructPair(ebigram,fbigram)
        eword_fbigram = ConstructPair(english,fbigram)
        ebigram_fword = ConstructPair(ebigram,foreign)

        ## bigram match and containments
        
        if bigram_pair.sides_match():
            features.add_binned(6)
            if tree != -1: features.add_binary(18,tree)
        elif bigram_pair.econtainsf():
            features.add_binned(7)
            features.add_binned(8)
        elif bigram_pair.fcontainse():
            features.add_binned(7)
            features.add_binned(9)

        ## e bigran and f unigram
        elif ebigram_fword.sides_match():
            features.add_binned(12)
            if tree != -1: features.add_binary(18,tree)
        elif eword_fbigram.sides_match():
            features.add_binned(13)

cdef inline long __ephrase_class(int class_id,int english_id,long num_phrases):
    """Compute the identifier associated with a english phrase and class 

    :param class_id: the identifier of the class 
    :param english_id: the identifier of the english phrase (-1 if unknown)
    :param num_phrases: the number of known phrases
    """
    if english_id == -1 or class_id == -1:
        return -1
    return (class_id*num_phrases)+english_id

cdef void __compute_bins(FeatureMap features):
    """Computes the binned features

    :param features: the feature map (which contains the bins)
    :rtype: None 
    """
    features.compute_neg_bins(np.array([1,3,6,7,8,9,12,13,16,44,36,24,25,57], ## feature list 
                                           dtype=np.int32),2.0)
    features.compute_neg_bins(np.array([23,49,47], ## feature list 
                                           dtype=np.int32),4.0)

cdef inline void __language_features(int lang_id,
                                        bint stop,
                                        int num_langs,
                                        int lpid,
                                        int lfid,
                                        bint pair_matches,
                                        int description,
                                        int lcid,
                                        FeatureMap features,
                ):
    """Features specific to output language type

    :param lang_id: language type identifier 
    :param num_langs: the total number of languages 
    :param lpid: the unigram pair id given the language type
    :param features: the features 
    """
    cdef int i

    if not stop and lpid >= 0:
        features.add_increm_binary(64,lpid)
        features.add_increm_binary(63,lfid)
    ## match per language 
    if pair_matches:
        features.add_increm_binary(68,lpid)
    ## class
    if lcid >= 0:
        features.add_increm_binary(70,lcid)
    if description != -1 and lang_id >= 0:
        features.add_increm_binary(69,lang_id)

        
cdef inline void __knowledge_features(int in_description,
                                          int dpid,
                                          int class_id,
                                          int cpid,
                                          int eid,
                                          int fid,
                                          bint match,
                                          int treepos,
                                          bint stop,
                                          bint in_alignment,
                                          FeatureMap features
                                          ):
    """Features related to background knowledge

    :param in_description: pair is in descriptions 
    :param dpid: the id of word/description pair 
    :param class_id: the identifier of the class (if exists) 
    :param cpid: the word/description pair for pair id
    :param eid: the english unigram identifier (-1 if unknown) 
    :param fid: the foreign unigram identifier (-1 if unknown) 
    :param match: binary switch indicating if the unigram pairs match 
    :param stop: english side is a stop word 
    :param in_alignment: indicates if pairs occur in viterbi alignment 
    :param features: the current feature map 
    """
    cdef bint description = True if in_description == 1 else False

    
    if description:
        features.add_binned(36)
        #features.add_binary(41,treepos)
        features.add_increm_binary(41,treepos)
        if dpid > -1: features.add_increm_binary(79,dpid)
        if match: features.add_binned(44)
        ## eside in description pairs
        if eid > -1 and not stop: features.add_binary(37,eid)
        if fid > -1 and not stop: features.add_binary(38,fid)
        ## viterbi alignment and in description 
        if is_alignment: features.add_binned(49)

    ## see also, or abstract, classes
    if class_id > -1 and not stop and eid > -1:
        features.add_binary(39,class_id)
        features.add_binary(40,cpid)
        if match: features.add_binary(48,class_id)
                
cdef int main_extractor(WordGraphDecoder decoder,
                            object config,
                            int[:] input_v,
                            unicode english,
                            int output_id,
                            int pos,
                            WordPhraseComponents word_phrase,
                            KnowledgeComponents knowledge,
                            RankComponents rank_info,
                            FeatureMap features
                            ) except -1:
    """The main feature extraction method 


    :param decoder: the underlying decoder model 
    :param input_v: the English side input vector 
    :param english: the surface form associated with the intput_v 
    :param output_id: the identifier associated with the output rep
    :param pos: the position of the output in the rank list 
    :param word_phrase: the word/phrase container object 
    :param knowledge: the knowledge container object 
    :param rank_info: the container object for rank/output information 
    :param features: the empty feature representation, to be filled here 
    """
    cdef int i,j,k
    cdef str lang = config.lang
    cdef int eid,fid,class_id,gclass,pair_id,dpid,cpid

    ## rank items
    cdef np.ndarray trees = rank_info.trees
    cdef int[:] output_v = rank_info.rank_item(output_id)
    cdef int[:] tree_pos = trees[output_id]
    cdef unicode foreign = rank_info.surface_item(output_id)

    ## vocab size
    cdef int fvsize = config.flen
    cdef int evsize = config.elen

    ### language information 
    cdef int lang_id = rank_info.language_id(output_id)
    cdef int num_langs = config.num_langs
    cdef int lpid,lfid,lcid 
    
    ### input/output length and list
    cdef int elen = input_v.shape[0]
    cdef int flen = output_v.shape[0]
    cdef list elist = english.split()
    cdef list flist = [u"<unknown>"]+foreign.split()
    cdef int ftree
    
    ## feature type information
    cdef bint phrase_f    = config.has_phrase
    cdef bint hiero_f     = config.has_hiero
    cdef bint compose_f   = config.has_compose
    cdef bint knowledge_f = config.has_knowledge

    ## unicode words 
    cdef unicode eword,fword,ebigram,fbigram
    cdef int num_classes = config.num_classes,prevtree

    ## switches
    cdef bint in_alignment
    cdef bint is_stop
    cdef bint pair_matches
    cdef bint pdbigram

    ## phrase heuristic
    cdef str heuristic = config.heuristic
    cdef SymAlign alignment
    cdef double[:,:] main_alignment
    
    ## word/phrase/knowledge information 
    cdef DescriptionWordPairs dwords = knowledge.descriptions
    cdef SparseDictWordPairs pairs = word_phrase.pairs
    cdef int num_pairs = pairs.size 
    
    cdef dict classes = rank_info.classes
    cdef dict class_sequence = classes.get(output_id,{})
    cdef int[:] cseq = np.ndarray((flen,),dtype=np.int32)
    cdef int[:] aseq = np.ndarray((flen,),dtype=np.int32)
    cdef int[:] dseq = np.zeros((elen,),dtype=np.int32)
    cdef bint already_checked

    ## word pair
    cdef PhrasePair word_component
    cdef bint has_multi

    ## check lengths

    if flen != len(flist):
        raise ValueError('Output vector does not match output string!')

    cseq[0] = -1
    aseq[0] = -1
    dseq[:] = -1

    ## rank position features (very important)
    __rank_features(pos,features)

    if lang_id > -1:
        __lang_rank_features(lang_id,pos,features,lang_id*15)

    # WORD LEVEL FEATURES  #
    ########################

    ## align the input and output
    alignment = decoder.align(output_v,input_v,heuristic)
    main_alignment = alignment.alignment
    gclass = -1
    has_multi = False

    ##input/output length features, class sequence information 
    for j in range(1,flen):
        fid = output_v[j]
        class_id = class_sequence.get(j-1,-1)
        if gclass != -1 and gclass != class_id: has_multi = True
        gclass  = class_id
        cseq[j] = class_id

    ## switch back it if has multiple classes
    if has_multi: gclass = -1

    ## go through the english words 
    for i in range(elen):
        eid     = input_v[i]
        eword   = elist[i]
        ebigram = __create_e_bigram(i,elist)

        ## go through the foreign words
        for j in range(1,flen):
            fid   = output_v[j]
            fword = flist[j]
            ftree  = tree_pos[j]
            fbigram = __create_f_bigram(j,flist)
            
            ## previous tree
            if j > 1: prevtree = tree_pos[j-1]
            else: prevtree = -1

            ## word pair information 
            word_component = ConstructPair(eword,fword)
            pair_id = pairs.find_identifier(fid,eid)
            
            ## pair features
            pair_matches = word_component.sides_match()
            is_stop = word_component.is_stop_word()

            ## word pairs and descriptions
            dpid = dwords.find_identifier(fword,eword)
            if dpid != -1: dseq[i] = 1
            
            ## f word and class features
            class_id = cseq[j]
            cpid = cat_index(eid,class_id,num_classes)
            
            ## viterbi alignment
            in_alignment = False
            if isfinite(main_alignment[i][j]): in_alignment = True

            ## word features
            __word_feat(word_component,
                            pair_matches,
                            eid,
                            fid,
                            pair_id,
                            is_stop,
                            ftree,
                            in_alignment,
                            features)

            ## bigram features
            __bigram_features(eword,fword,
                                ebigram,
                                fbigram,
                                ftree,
                                prevtree,
                                False,
                                fid,
                                features)

            ## knowledge related features
            if knowledge_f:
                __knowledge_features(dseq[i],dpid,
                                         class_id,
                                         cpid,
                                         eid,
                                         fid,
                                         pair_matches,
                                         is_stop,
                                         ftree,
                                         in_alignment,
                                         features)

            ## language specific features (if available)
            if lang_id > -1:
                lpid = (lang_id*num_pairs)+pair_id if pair_id >= 0 else -1
                lfid = (lang_id*fvsize)+fid if fid >= 0 else -1
                lcid = (lang_id*num_classes)+cpid if cpid >= 0 else -1

                    
                ## language features 
                __language_features(lang_id,
                                        is_stop,
                                        num_langs,
                                        lpid,
                                        lfid,
                                        pair_matches,
                                        dseq[i],
                                        lcid,
                                        features)

    ## general language related features

    # PHRASE FEATURES      #
    ########################
    
    if phrase_f:
        ## performs the phrase search
        phrase_search(config,alignment,
                          word_phrase,
                          elen,
                          flen,
                          elist,
                          flist,
                          dseq,
                          cseq,
                          tree_pos,
                          gclass,
                          lang_id,
                          features)

    ## compute binned features
    __compute_bins(features)

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

cdef inline void __phrase_features(PhrasePair pair,
                                      int pid,
                                      int epid,
                                      int fpid,
                                      int ftree,
                                      int elen,
                                      int flen,
                                      bint match,
                                      bint stop,
                                      bint indescr,
                                      long clpair_id,
                                      long lang_start,
                                      int lang_id,
                                      FeatureMap features
                                      ):
    """Add phrase related features to the features


    :param pair: the phrase pair object 
    :param pid: the phrase identifier (-1 if unknown) 
    :param epid: the english phrase identifier (-1 if unknown) 
    :param fpid: the foreign phrase identifier (same as above) 
    :param elen: the lenght of the english phrase 
    :param flen: the length of the foreign phrase 
    :param match: the phrases match 
    :param stop: the english side is a stop word only 
    :param indescr: the english starts and stops with words in descriptions 
    :param clpair_id: the class phrase pair identifier 
    """
    cdef bint known = True if pid > -1 else False
    cdef bint clpair = True if clpair_id > -1 else False
    cdef bint great_one = True if (elen > 1 or flen > 1) else False
    cdef bint tree_match = True if ftree != -1 else False
    cdef int word_overlap

    ## do not consider phrases 
    if (not great_one or stop) and not match:
        return
    
    ## matching and overlapping 
    word_overlap = pair.word_overlap()
    if match:
        features.add_binned(24)
        if tree_match: features.add_binary(31,ftree)
        phrase_length_computation(elen,27,features)
        phrase_length_computation(flen,28,features)
        if lang_id >= 0:
            features.add_increm_binary(74,lang_id)
                            
    elif pair.econtainsf() or pair.fcontainse():
        features.add_binned(25)
        phrase_length_computation(elen,51,features)
        phrase_length_computation(flen,52,features)
        if tree_match: features.add_binary(26,ftree)
        if lang_id >= 0:
            features.add_increm_binary(75,lang_id)
        
    elif word_overlap > 0:
        if tree_match: features.add_binary(26,ftree)
        phrase_length_computation(elen,51,features)
        phrase_length_computation(flen,52,features)
        if lang_id >= 0:
            features.add_increm_binary(75,lang_id)

    ## known phrases
    if known:
        #features.add_binary(22,pid)
        features.add_increm_binary(22,pid)
        features.add_binned(23)
        ## individual known phrases
        # features.add_binary(29,epid)
        # features.add_binary(30,fpid)
        features.add_increm_binary(29,epid)
        features.add_increm_binary(30,fpid)
        ## sizes of known phrases
        phrase_length_computation(elen,53,features)
        phrase_length_computation(flen,54,features)

        ## language specific 
        if lang_start >= 0:
            features.add_increm_binary(72,lang_start+pid)
        
    if clpair:
        #features.add_binary(42,clpair)
        features.add_increm_binary(42,clpair)
    if indescr:
        features.add_binned(47)

cdef inline void __hiero_features(unicode lhs, ## lhs side rule 
                              unicode eside,unicode fside,## phrase content
                              HieroPhraseTable hiero_rules, ## hiero grammar
                              FeatureMap features,
                              bint unique,
                              int class_id,
                              int lang_id,
         ):
    """Features related to lexical (i.e., without non-terminal) hierarchical phrase rules 

    -- note these rules are essentially phrase rule, with more information about tree, 
    therefore we do not extract so much information from them. 

    :param lhs: the left hand side of the rule 
    :param eside: the english side of the rule 
    :param fside: the foreign side of the rule 
    :param hiero_rules: the hiero grammar 
    :param features: the current feature set 
    """
    cdef HieroRule rule = hiero_rules.create_rule(eside,fside,lhs)
    cdef int identifier = rule.rule_number
    cdef int eknown = rule.eid
    cdef int fknown = rule.fid
    cdef int nume = hiero_rules.elen
    cdef int num_hiero = hiero_rules.num_phrases
    cdef int classhid
    cdef long lang_start = -1 if lang_id < 0 else num_hiero*lang_id

    ## lhs identifier
    cdef dict lhs_lookup = hiero_rules.lhs_lookup
    cdef int lhs_id = lhs_lookup.get(lhs,-1)

    ## hiero start

    if identifier != -1:
        #features.add_binary(32,identifier)
        features.add_increm_binary(32,identifier)
        if unique:
            # features.add_binary(35,fknown)
            # features.add_binary(34,eknown)
            features.add_increm_binary(35,fknown)
            features.add_increm_binary(34,eknown)
        if lang_start >= 0:
            features.add_increm_binary(73,lang_start+identifier)
            features.add_increm_binary(76,(lang_id*nume)+eknown)
            features.add_increm_binary(77,(lang_id*numf)+fknown)
            
    if lhs_id != -1:
        features.add_binary(55,lhs_id)

    ## english side and abstract class
    if eknown != -1 and class_id != -1:
        classhid = (class_id*nume)+eknown
        features.add_binary(58,classhid)
        
                
cdef int phrase_search(object config,
                          SymAlign alignment,
                          WordPhraseComponents word_phrase,
                          int elen,
                          int flen,
                          list einput,
                          list finput,
                          int[:] descriptions,
                          int[:] classes,
                          int[:] treepos,
                          int gclass,
                          int lang_id,
                          FeatureMap features
                          ) except -1:
    """Search for phrase occurrences """
    cdef Phrases phrases_found = alignment.extract_phrases(7)
    cdef int[:,:] phrase_loc = phrases_found.phrases
    cdef int num_phrases = phrase_loc.shape[0]
    cdef int i,j,k,w
    cdef unicode english_side,foreign_side,foreing_modified
    cdef str lang = config.lang
    cdef long clpair

    ## example phrases 
    cdef PhrasePair phrase_instance
    cdef bint hiero = config.has_hiero

    ## phrase container
    cdef SimplePhraseTable phrases = word_phrase.phrases
    cdef int num_ephrases = phrases.elen
    cdef int num_fphrases = phrases.flen
    cdef int table_size = phrases.num_phrases
    ## offset for language phrases 
    cdef long lang_start = -1 if lang_id < 0 else lang_id*table_size

    ## phrase properties
    cdef bint phrase_matches,is_stop
    cdef int fsidelen, esidelen
    cdef int class_id,aspair,ftree
    cdef int sfindescr

    ## see also classes
    cdef int num_classes = config.num_classes

    ## with alignment
    cdef int[:] e_aligned,f_aligned

    ## phrase properties
    cdef int phrase_id,english_pid,foreign_pid
    cdef int fstart,fend,estart,eend

    ## hiero stuff
    cdef HieroPhraseTable hiero_rules = word_phrase.hiero
    cdef tuple tuple1
    cdef unicode tlh1,lhs
    cdef int edist,fdist
    cdef dict rule_map = hiero_rules.glue
    cdef list echart
    cdef dict lookup_table = {}
    cdef int hiero_size = hiero_rules.num_phrases

    ## alignment
    e_aligned = np.zeros((elen,),dtype=np.int32)
    f_aligned = np.zeros((flen,),dtype=np.int32)
    echart = [[u'' for _ in range(elen+1)] for _ in range(elen)]


    for i in range(num_phrases):

        #conditions
        class_id  = -1
        clpair    = -1
        sfindescr = False
        aspair    = -1
        ftree     = -1

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
        ## foreign side w/o @ meta symbol
        foreign_modified = re.sub(r'\@.+$','',foreign_side.strip()).strip()

        ## length of foreign phrase 
        fsidelen = fend+1 - fstart
        esidelen = eend+1 - estart

        ## phrase instance 
        phrase_instance = phrases.create_pair(english_side,foreign_side)

        ## idenfitifiers (-1 if unknown)
        phrase_id = phrase_instance.num
        english_pid = phrase_instance.eid
        foreign_pid = phrase_instance.fid
        
        ## phrase match, is a stop word  
        phrase_matches = True if phrase_instance.sides_match() or english_side == foreign_modified else False
        is_stop = True if phrase_instance.is_stop_word() else False
        
        ## phrase consistent with tree? (should be larger than 1)
        if treepos[fstart] == treepos[fend]:
            ftree = treepos[fstart]
        
        ## in a class
        if classes[fstart] > -1 and classes[fstart] == classes[fend]:
            class_id =  classes[fstart]
            clpair = __ephrase_class(class_id,english_pid,num_ephrases)

        ## e side begins and ends in descriptions
        if descriptions[estart] == 1 and descriptions[estart] == descriptions[eend]:
            sfindescr = True

        __phrase_features(phrase_instance,
                              phrase_id,
                              english_pid,
                              foreign_pid,
                              ftree,
                              esidelen,
                              fsidelen,
                              phrase_matches,
                              is_stop,
                              sfindescr,
                              clpair,
                              lang_start,
                              lang_id,
                              features)

        ## store prhases for later hiero searc
        if hiero:

            tuple1 = (str(treepos[fstart]),str(treepos[fend]))
            tlhs1  = rule_map.get(tuple([str(treepos[fstart])]),u'')
            edist = eend - estart
            fdist = fend - fstart

            if estart == eend:
                lhs = tlhs1

                ## single tree span 
                if ftree != -1 and lhs:
                    echart[estart][eend+1] = lhs
                    lookup_table[(estart,eend+1)] = (fstart,fend+1)

                    ## add this rule 
                    #__hiero_features(lhs,english_side,foreign_side,hiero_rules,features,False,gclass,lang_id)
                    

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

                    #__hiero_features(lhs,english_side,foreign_side,hiero_rules,features,False,gclass,lang_id)

                ## can be fixed with one shift to the right
                elif not fend-1 <= fstart and treepos[fend-1] == ftree and lhs:
                    lookup_table[(estart,eend+1)] = (fstart,fend)
                    echart[estart][eend+1] = lhs

                    __hiero_features(lhs,english_side,foreign_side,hiero_rules,features,True,gclass,lang_id)

            ## long spans with abstract lhs
            elif edist <= 3 and ftree != -1 and tlhs1:
                echart[estart][eend+1] = tlhs1
                lookup_table[(estart,eend+1)] = (fstart,fend+1)

                #__hiero_features(tlhs1,english_side,foreign_side,hiero_rules,features,False,gclass,lang_id)

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
                lookup_table[(estart,eend+1)] = (fstart,fend+1)
                echart[estart][eend+1] = lhs
                ## lexical hiero rules 
                #__hiero_features(lhs,english_side,foreign_side,hiero_rules,features,False,gclass,lang_id)

    if hiero:
        hiero_search(config,
                         hiero_rules,
                         elen,
                         flen,
                         finput,
                         einput,
                         echart,
                         lookup_table,
                         e_aligned,
                         f_aligned,
                         gclass,
                         treepos,
                         lang_id,
                         features)

## auxiliary hiero functions

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

cdef inline void __hiero_gap_rule(unicode lhs,
                                      unicode eside,
                                      unicode fside,
                                      HieroPhraseTable hiero_rules,
                                      bint reorder,
                                      int class_id,
                                      str lang,
                                      int lang_id,
                                      FeatureMap features,
                                      ):
    """Recognize a hiero gap and add to features 


    :param lhs: the lhs of the hiero rule 
    :param eside: the english side 
    :param fside: the foreign side 
    :param hiero_rules: the hiero container object 
    :param reorder: indicates if the rule has re-ordering 
    :param class_id: the class identifier associated with output 
    :param lang: the english side language 
    :param features: the current feature map
    """
    cdef int i
    cdef HieroRule rule = hiero_rules.create_rule(eside,fside,lhs,lang=lang)
    cdef int identifier = rule.rule_number
    cdef int eknown = rule.eid
    cdef int fknown = rule.fid
    cdef int num_hiero = hiero_rules.num_phrases
    cdef classhid

    ## properties of container
    cdef int nume = hiero_rules.elen
    cdef int numf = hiero_rules.flen

    ## check if english side is a stop word
    cdef bint is_stop = rule.is_stop_word()
    cdef bint only_terminals = rule.only_terminals()
    cdef bint eterm = rule.left_terminal_only()
    cdef bint fterm = rule.right_terminal_only()

    ## lhs identifier
    cdef dict lhs_lookup = hiero_rules.lhs_lookup
    cdef int lhs_id = lhs_lookup.get(lhs,-1)

    if identifier != -1 and not only_terminals and not is_stop:
        features.add_increm_binary(32,identifier)
        features.add_binned(23)
        if lang_id >= 0:
            features.add_increm_binary(73,(lang_id*num_hiero)+identifier)
            features.add_increm_binary(76,(lang_id*nume)+eknown)
            features.add_increm_binary(77,(lang_id*numf)+fknown)

        ## classes
        if class_id != -1:
            classhid = (class_id*num_hiero)+identifier
            features.add_binary(46,classhid)

    ## lhs type
    if lhs_id != -1:
        features.add_binary(55,lhs_id)

    ## reordering
    if reorder:
        features.add_binned(57)
        if lhs_id != -1:
            features.add_binary(56,lhs_id)
        if lang_id >= 0:
            features.add_binary(78,lang_id)
            
    # ## english side
    if eknown != -1 and not eterm and not is_stop:
        features.add_binary(34,eknown)
        ## abstract class and english side
        classhid = (class_id*nume)+eknown
        if class_id != -1: features.add_binary(58,classhid)

    if fknown != -1 and not fterm:
        features.add_binary(35,fknown)

cdef inline int hiero_search(object config,
                                HieroPhraseTable rules,
                                int elen,
                                int flen,
                                list finput, list einput,
                                list echart,
                                dict lookup_table,
                                int[:] e_aligned,int[:] f_aligned,
                                int gclass,int[:] trees,
                                int lang_id,
                                FeatureMap features,
                                )  except -1:
    """A simplified chart search procedure for finding hiero rules 


    :param config: the main configuration 
    :param elen: the length of the english input 
    :param flen: the length of the foreign output 
    :param lookup_table: the lookup table for existing spans 
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

    ## hiero
    cdef int num_hiero = rules.num_phrases
    cdef dict rule_map = rules.glue

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

            for mid in range(start+1,end):
                span1 = echart[start][mid]
                span2 = echart[mid][end]

                ## no patterns found 
                if (not span1 or not span2) or (span1,span2) not in rule_map:
                    continue

                lhs = rule_map[(span1,span2)]
                fs1,fe1 = lookup_table[(start,mid)]
                fs2,fe2 = lookup_table[(mid,end)]
                if (fs1,fe1) == (fs2,fe2): continue
                fstart = fs1 if fs1 < fs2 else fs2
                fend = fe1 if fs1 > fs2 else fe2

                ## see if there is a middle gap
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
                __hiero_gap_rule(lhs,espan,fspan,rules,reordering,gclass,lang,lang_id,features)
                
                if (mid - start) <= 2:
                    espan = to_unicode(' '.join([einput[w] for w in range(start,mid)]+["[%s_2]" % span2]))
                    __hiero_gap_rule(lhs,espan,fspan,rules,reordering,gclass,lang,lang_id,features)

                    if reordering:
                        __hiero_gap_rule(lhs,espan,u"[%s_2] [%s_1]" % (span2,span1),rules,reordering,gclass,lang,lang_id,features)
                    else:
                        __hiero_gap_rule(lhs,espan,u"[%s_1] [%s_2]" % (span1,span2),rules,reordering,gclass,lang,lang_id,features)

                if (end - mid) <= 2:
                    espan = to_unicode(' '.join([u"[%s_1]" % span1]+[einput[w] for w in range(mid,end)]))
                    __hiero_gap_rule(lhs,espan,fspan,rules,reordering,gclass,lang,lang_id,features)

                    if reordering:
                        __hiero_gap_rule(lhs,espan,u"[%s_2] [%s_1]" % (span2,span1),rules,reordering,gclass,lang,lang_id,features)
                    else:
                        __hiero_gap_rule(lhs,espan,u"[%s_1] [%s_2]" % (span1,span2),rules,reordering,gclass,lang,lang_id,features)
    
## CLI INFORMATION

def params():
    """Graph decoder extractors

    """
    group = {"GraphExtractor" : "Graph Feature Extractor settings"}

    options = [
        ("--atemplates","atemplates","lex+phrase+align+knowledge+hiero+compose+paraphrase","str",
            "types of aligner templates [default=lex+phrase+hiero+knowledge+compose]","GraphExtractor"),
        ("--hierogrammar","hierogrammar","","str",
            "location of the hiero grammar [default='']","GraphExtractor"),
        ("--gluegrammar","gluegrammar","","str",
            "location of the glue grammar [default='']","GraphExtractor"),
        ("--ispolyglot","ispolyglot",False,"bool",
            "Model is a polyglot model [default=False]","GraphExtractor"),
        ("--store_feat","store_feat",False,"bool",
            "extract features only once to file  [default=False]","GraphExtractor"),
        ("--concur_extract","concur_extract",False,"bool",
            "Concurrently extract features over several asynchronous processed [default=False]","GraphExtractor"),
        ("--num_extract","num_extract",2,int,
            "The number of asynchronous processed to run [default=2]","GraphExtractor"),
        ("--feat_blocks","feat_blocks","","str",
            "Features to block [default='']","GraphExtractor"),
        ("--train_ranks","train_ranks","","str",
            "Path to train ranks (if computed offline) [default='']","GraphExtractor"),
        ("--valid_ranks","train_ranks","","str",
            "Path to validation ranks (if computed offline) [default='']","GraphExtractor"),
    ]
        
    return (group,options)

def argparser():
    """Return an aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    
    usage = """python -m zubr graphextractor [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

