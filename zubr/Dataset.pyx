# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package. 

author : Kyle Richardson

An abstract Dataset class and related utilities for representing 
training and testing data

"""

import os
import pickle
import copy
import time
import codecs
import logging
import numpy as np
cimport numpy as np
from zubr.util.aligner_util import get_lex,get_rdata,load_aligner_data
from zubr.ZubrClass cimport ZubrSerializable
from zubr.util.dataset_util import score_rank,compare_ranks,score_poly,score_mono,compare_multi_ranks

cdef class Pair:

    """Generic pair of two (numpy array) items"""

    def __init__(self,e,f):
        """Construcs a pair instance for e and f

        :param e: e array input
        :param f: f array input
        """
        self.e = e
        self.f = f

    property elen:
        """length attribute of e item"""

        def __get__(self):
            """return the length of e
            
            :rtype: int
            """
            return <int>self.e.shape[0]

    property flen:
        """length attribute of f item"""

        def __get__(self):
            """return the length of f

            :rtype: int
            """
            return <int>self.f.shape[0]

    property lmatch:
        """check if lengths match"""
    
        def __get__(self):
            """Determines is the lengths match

            :rtype: bool
            """
            return <bint>(self.f.shape[0] == self.e.shape[0])
        
    def __reduce__(self):
        return Pair,(self.e,self.f)

cdef class RankPair:

    """Represents a training instance, encoded example with 
    pointer to the target representation identifier """

    def __init__(self,en,indx,global_id,surface=[],lang=None):
        """Creates a RankPair instance

        :param en: the example vector representation 
        :param indx: the gold id identifier 
        :param global_id: the global id in overall dataset
        """
        self.en        = en
        self.rep_id    = indx
        self.global_id = global_id
        self.surface   = surface
        self.lang      = lang
        
    property elen:
        """Elen is the length of instance input"""
    
        def __get__(self):
            """return the length of e

            :rtype: int 
            """
            return <int>self.en.shape[0]

    def __reduce__(self):
        ## pickle implementation
        return RankPair,(self.en,self.rep_id,self.global_id)
    

cdef class Data(ZubrSerializable):
    """Base class for representing different types of data"""

    property size:
        """The size of the encoded data"""
    
        def __get__(self):
            """Returns the size 
            
            :rtype: int
            """
            raise NotImplementedError

    property is_empty:
        """Determine if the data is empty or not"""
    
        def __get__(self):
            """Returns whether empty or not 

            :rtype: bint
            """
            raise NotImplementedError

    
cdef class EmptyDataset(Data):

    """An empty dataset"""
    
    property size:
    
        """Determines the size of empty dataset, always zero"""

        def __get__(self):
            """Get the emptydataset size

            :returns: zero for empty 
            :rtype: int
            """
            return 0

    property is_empty:
        """Determines is dataset is emptu or not"""
        
        def __get__(self):
            """Returns true (is always empty)

            :returns: True
            """
            return True

cdef class Dataset(Data):

    """Class for representing a given datasets"""


    property size:
        """The size attribute size is the number of examples"""
        
        def __get__(self):
            """return the size
            
            :rtype: int 
            """
            return <int>self._size

    property index:
        """The index is the position of the current item"""
        
        def __get__(self):
            """return the current point in training 
            
            :rtype: int 
            """
            return <int>self._dataset_order[self._index]

    property is_empty:
        """A dataset is empty is it has no examples"""
    
        def __get__(self):
            """Gets whether the dataset is empty

            :rtype: bool
            """
            return <bint>(self._size <= 0)

    property multilingual:
        """A dataset is empty is it has no examples"""
    
        def __get__(self):
            """Gets whether the dataset is empty

            :rtype: bool
            """
            raise NotImplementedError 
        
    def py_shuffle(self):
        ## python call to shuffle method (currently for testing)
        self.shuffle()
        
    def py_next(self):
        ## python call to next_ex method (currently for testing)
        return self.next_ex()
        
    cdef void shuffle(self):
        """shuffle the order of the dataset (e.g. for online learning)

        :rtype: None
        """
        np.random.seed(10)
        np.random.shuffle(self._dataset_order)
        self.logger.info('Shuffled dataset order...')
        self._index = 0

    cpdef RankPair get_item(self,int index):
        """get a training pair by index

        :param index: index of the item
        :returns: a pairt of examples
        """
        raise NotImplementedError

    cdef RankPair next_ex(self):
        """return the next example in training data

        :raises IndexError: when dataset has already been enumerated 
          (means that a shuffle should have occurred)
        """
        raise NotImplementedError
        
    def print_data(self):
        """print out the dataset"""
        raise NotImplementedError
    
cdef class RankDataset(Dataset):
    """dataset with list of items to rank"""

    def __init__(self,en,en_orig,rank_index,langs=None,trace=True):
        if rank_index.shape[0] != en.shape[0]:
            raise ValueError('gold rank index has different size')

        self.en = en
        self.en_original = en_orig
        self._size = self.en.shape[0]
        self._index = 0
        self._dataset_order = np.array([i for i in range(self._size)])
        self.rank_index = rank_index

        ## language information (if available) 
        if langs is None:
            if trace: self.logger.info('loaded monolingual dataset of size: %d' % self._size)
            self.langs = np.empty(0)
        else:
            if trace: self.logger.info('loaded multingual dataset of size: %d' % self._size)
            self.langs = langs

    cpdef RankPair get_item(self,int index):
        """get a training pair by index

        :param index: index of the item
        :returns: a pairt of examples
        """
        lang = None if not self.multilingual else self.langs[index]
        return RankPair(self.en[index],self.rank_index[index],index,
                            surface=np.unicode(self.en_original[index]),lang=lang)

    cdef RankPair next_ex(self):
        """return the next example in training data

        :raises IndexError: when dataset has already been enumerated 
          (means that a shuffle should have occurred)
        """
        if self._index >= self._size:
            self.logger.info('Resetting iterator (should you be shuffling?)...')
            self._index = 0

        new_index = <int>self._dataset_order[self._index]
        self._index += 1
        lang = None if not self.multilingual else self.langs[new_index]
        return RankPair(self.en[new_index],self.rank_index[new_index],new_index,
                        surface=np.unicode(self.en_original[new_index]),lang=lang)


    def split_dataset(self,wdir,dtype,jname,num_splits):
        """Chops up main dataset into several pieces, and backs up these pieces

        This is primarily used for running parallel processes 

        :param wdir: the current working directory 
        :param dtype: the type of data 
        :param jname: the name of job directory
        :pararm num_splits: the number of splits/jobs to perform  
        """
        piece_size = self._size/num_splits
        remainder = self._size % num_splits
        jobs = os.path.join(wdir,jname)
        #last = num_splits
        last = 0
        if not os.path.isdir(jobs): raise ValueError('Uknown jobs directory: %s' % jobs)
        no_langs = (self.langs.shape[0] == 0)

        for i in range(num_splits):
            dout = os.path.join(jobs,"job_%d/%s.data" % (i,dtype))
            text_out = os.path.join(jobs,"job_%d/%s_text.txt" % (i,dtype))
            if i != (num_splits - 1):
                langs = None if no_langs else self.langs[last:last+piece_size]
                RankDataset(self.en[last:last+piece_size],self.en_original[last:last+piece_size],
                                self.rank_index[last:last+piece_size],langs=langs).dump(dout)
                last += piece_size
                        
            else:
                langs = None if no_langs else self.langs[last:last+piece_size+remainder]
                RankDataset(self.en[last:last+piece_size+remainder],self.en_original[last:last+piece_size+remainder],
                                self.rank_index[last:last+piece_size+remainder],langs=langs).dump(dout)

    @classmethod
    def single_dataset(cls,encoded,surface,gold=-1):
        """Builds a dataset from a single data instance

        :param encoded: the encoded representation 
        :param surface: the surface form
        :param gold: the gold identifier (if exists) 
        """
        return cls(np.array([encoded]),
                       np.array([surface],dtype=np.unicode),
                       np.array([-1],dtype=np.int32),trace=False)
        
    def __reduce__(self):
        #return RankDataset,(self.en,self.en_original,self.rank_index)
        return RankDataset,(self.en,self.en_original,self.rank_index,self.langs)

    ## properties
    property multilingual:
        """A dataset is empty is it has no examples"""

        def __get__(self):
            """Gets whether the dataset is empty

            :rtype: bool
            """
            return not (self.langs.shape[0] == 0)


EMPTY_RANK = RankDataset(np.empty((0,)),np.empty((0,)),np.empty((0,)))

################### rank evaluator

cdef class RankEvaluator:
    """Stores ranks for a given dataset and evaluates and computes scores"""

    def __init__(self,data_size,rank_size,rtype):
        """Initializes a rank evaluator instance

        :param data_size: the size of the dataset 
        :param rank_size: the rank size 
        :param rtype: the type of evaluation 
        """
        self.size      = data_size
        self.rank_size = rank_size
        self.rtype     = rtype

#################### rank storage

cdef class RankStorage(ZubrSerializable):

    "Stores the offline ranks for a base model"

    def __init__(self,ranks,gold,other_gold={}):
        """Initialized a rank vector 

        :param ranks: the ranks for each item in dataset 
        :param gold_pos: the position of gold in ranks
        :param other_gold: the position of other items in rank that evaluate to true
        """
        self.ranks = ranks
        self.gold_pos  = gold
        self.other_gold = other_gold 

    @classmethod
    def load_empty(cls,data_size,rank_size):
        """load an item from size 

        :param dsize: the data size
        :param rsize: the rank size 
        """
        ranks = np.ndarray((data_size,rank_size),dtype='int32')
        gold_pos = np.ndarray((data_size,),dtype='int32')
        ranks.fill(-1)
        gold_pos.fill(-1)

        return cls(ranks,gold_pos)

    property rank_size:
        """Determine the size of the ranks"""

        def __get__(self):
            cdef np.ndarray[ndim=2,dtype=np.int32_t] oranks = self.ranks
            return oranks.shape[1]-1
        
    @classmethod
    def load_from_file(cls,file_path,dsize,rsize):
        """Create a rank storage from file

        :param file_path: the path to the file 
        :param dsize: the size of the dataset 
        :param rsize: the total number of rank items or beam size 
        """
        ## initialize
        ranks = np.ndarray((dsize,rsize),dtype='int32')
        gold_pos = np.ndarray((dsize,),dtype='int32')
        ranks.fill(-1)
        gold_pos.fill(-1)
        total_number = 0
        
        with open(file_path) as my_ranks:
            for line in my_ranks:
                line = line.strip()
                number,gold_id,number_list = line.split('\t')
                number  = int(number)
                gold_id = int(gold_id)
                contains_gold = False
                total_number += 1

                ## go through the available ranks 
                for k,rank_item in enumerate(number_list.split()):
                    rank_item = int(rank_item)
                    ranks[number][k] = rank_item
                    if rank_item == gold_id:
                        gold_pos[number] = k
                        contains_gold = True

                ## put in rsize as gold position if not in beam
                if not contains_gold:
                    gold_pos[number] = (rsize - 1)
                    ranks[number][rsize-1] = gold_id
                    
        instance = cls(ranks[:total_number],gold_pos[:total_number])
        instance.logger.info('Loaded a rank storage item, with size (%d,%d)' % (total_number,rsize))
        return instance
    
    ## backup protocol 
    
    def backup(self,wdir,name='valid'):
        """Backup the rank storage items to file using numpy backup

        :param wdir: the working directory 
        :param name: the name of the type of data 
        :rtype: None 
        """
        self.logger.info('Backing up the storage for <%s>..' % name)
        stime = time.time()
        sfile = os.path.join(wdir,"%s_storage" % (name))
        np.savez_compressed(sfile,self.ranks,self.gold_pos,self.other_gold)
        self.logger.info('Backed up in %s seconds' % (time.time()-stime))

    @classmethod
    def load_backup(cls,config,name='valid'):
        """Load the backup from file to create an instance 

        :param config: the global configuration 
        """
        sfile = os.path.join(config.dir,"%s_storage.npz" % name)
        other_gold = False

        ## file archive 
        archive = np.load(sfile)
        ranks = archive["arr_0"]
        gold  = archive["arr_1"]
        other = archive["arr_2"].item()

        ## create instance
        instance = cls(ranks,gold,other)
        instance.logger.info('Other gold=%s' % str(other != {}))
        return instance
    

    def compute_score(self,path,dtype='train'):
        """Computes the base score

        :param path: the path to the working directory
        :rtype: None 
        """
        st = time.time()
        score_rank(path,self.ranks,self.gold_pos,dtype)
        self.logger.info('Computed <%s> score in %f seconds' %\
                             (dtype,time.time()-st))

    def compute_mono_score(self,path,k,dtype='train'):
        """Compute score for non-polyglot model

        note: there is large overlap with the function above, 
        it should be merged, but I didn't want to break anything. The 
        difference relates to computing the MRR when the beam doesn't 
        contain the target representation. 

        :param path: the path to print the result 
        :param k: the size to print 
        :param dtype: the type of data 
        """
        st = time.time()
        score_mono(path,self.ranks,self.gold_pos,k,dtype)
        self.logger.info('Computed <%s> score in %f seconds' %\
                             (dtype,time.time()-st))

    def compute_poly_score(self,path,langs,rmap,k,dtype='train',exc=False):
        """Compute score for where the output is multiple languages 

        :param path: the path to print the results 
        :param langs: the listing of the languages in the rank list 
        :param rmap: the rank map (used for determing the size of each language rank list)
        """
        st = time.time()
        score_poly(path,langs,self.ranks,rmap,self.gold_pos,k,dtype,exc=exc)
        self.logger.info('Computer <%s> score in %f seconds' % (dtype,time.time()-st))
        
    def compare_ranks(self,path,new_ranks):
        """Compare baseline ranks to new ranks

        :param path: path to working directory 
        :param new_ranks: the new ranks to compare to
        """
        pass

    def __reduce__(self):
        return RankStorage,(self.ranks,self.gold_pos)

    cdef np.ndarray find_ranks(self,int indx,str ttype,int beam):
        """Find the stored ranks for a given index

        -- if ttype is set to ``train``, it removes the gold item 

        :param indx: the index of the target rank item
        :param ttype: the current data type  
        :param beam: the number of rank items to return
        """
        cdef int[:] gold_pos = self.gold_pos
        cdef int i,k,gold
        cdef np.ndarray[ndim=2,dtype=np.int32_t] oranks = self.ranks
        cdef np.ndarray[ndim=1,dtype=np.int32_t] lrank = oranks[indx][:beam]

        if ttype == 'train':
            gold = gold_pos[indx]
            return  np.array([i for k,i in enumerate(lrank) if k != gold],dtype=np.int32)
        
        return lrank

    cdef int gold_position(self,int indx):
        """Find the rank gold position for a given index

        :param indx: the target index 
        :returns: the integer location of gold in list (not the value)
        :rtype: int
        """
        cdef int[:] gold_ids = self.gold_pos
        return <int>gold_ids[indx]

    cdef int gold_value(self,int indx):
        """Returns the gold value associated with an index

        :param indx: the index
        :returns: the gold identifier
        :rtype: int
        """
        cdef int[:] gold_pos = self.gold_pos
        cdef int[:,:] oranks = self.ranks
        return oranks[indx][gold_pos[indx]]

    cdef np.ndarray empty_rank_list(self):
        """Creates an empty rank list (e.g., for comparison to current)

        :returns: an empty copy of the storage ranks 
        """
        cdef np.ndarray[np.int32_t,ndim=2] new_ranks,ranks = self.ranks
        new_ranks = copy.deepcopy(ranks)
        new_ranks.fill(-1)
        
        return new_ranks
    
    property is_empty:
        "Determines if storage is empty"

        def __get__(self):
            """Return bool of whether storage is empty
            
            :rtype: bool
            """
            return <bint>(self.ranks.shape[0] <= 0)

EMPTY_STORAGE = RankStorage.load_empty(0,0)

### RANK comparison class 

cdef class RankComparison(ZubrSerializable):
    """Stores a rank storage and a new ranked list for comparison"""

    def __init__(self,storage,beam=-1):
        """Initializes a rank comparison instance using an example storage

        :param storage: a current list of examples with ranks 
        :type storage: RankStorage
        :param beam: the size of beam used in new ranking (if all, is -1)
        """
        self.storage = storage
        self.beam = beam
        self.new_ranks = self.storage.empty_rank_list()

    cdef np.ndarray rank_example(self,int i):
        """Returns an empty set of ranks given an index i

        :param i: the index 
        """
        cdef np.ndarray[np.int32_t,ndim=2] total = self.new_ranks
        return <np.ndarray>total[i]

    cdef np.ndarray old_ranks(self,int i):
        """Returns an empty set of ranks given an index i

        :param i: the index 
        """
        cdef RankStorage storage = self.storage
        cdef np.ndarray[np.int32_t,ndim=2] ranks = storage.ranks
        return <np.ndarray>ranks[i]

    cdef RankScorer evaluate(self,str ttype,str wdir,int it=-1,double ll=-1,bint debug=True):
        """Evaluate after a new ranking takes place

        :param ttype: the type of testing 
        :param wdir: the working directory 
        :param it: the number of training iterations performed before testing
        :param ll: the data log likehood (if available)
        """
        cdef RankStorage storage = self.storage
        cdef RankScorer r
        cdef np.ndarray[np.int32_t,ndim=2] ranks = storage.ranks
        cdef np.ndarray[np.int32_t,ndim=1] gold = storage.gold_pos
        cdef np.ndarray[np.int32_t,ndim=2] new_ranks = self.new_ranks

        st = time.time()
        r = compare_ranks(ranks,new_ranks,gold,ttype,it,ll,wdir,RankScorer,storage.other_gold)
        ## print backup of ranks (for later use with )

        if debug:
            self.logger.info('Computed <%s> score in %f seconds' % (ttype,time.time()-st))
        return r

    cdef RankScorer multi_evaluate(self,RankDataset dataset,str ttype,str wdir,int it=-1,double ll=-1,bint debug=True):
        """Evaluate a multilingual dataset after reranking takes place 

        :param ttype: the type of testing 
        :param wdir: the working directory 
        :param it: the number of training iterations 
        :param ll: the data log likelihood (if available)
        """
        cdef RankStorage storage = self.storage
        cdef RankScorer r
        cdef np.ndarray[np.int32_t,ndim=2] ranks = storage.ranks
        cdef np.ndarray[np.int32_t,ndim=1] gold = storage.gold_pos
        cdef np.ndarray[np.int32_t,ndim=2] new_ranks = self.new_ranks
        cdef np.ndarray langs = dataset.langs

        stime = time.time()        
        r = compare_multi_ranks(langs,ranks,new_ranks,gold,ttype,it,ll,wdir,RankScorer,storage.other_gold)

        if debug:
            self.logger.info('Computed <%s> score in %f seconds' % (ttype,time.time()-stime))
        return r

### SCORING UTILITIES

SCORE = {
    
}

cdef class Scorer:
    """Base class for scoring items"""

    def __richcmp__(self,other,int opt):
        ## score comparison
        if opt == 0:
            return self.less_than(other)
        elif opt == 4:
            return self.greater_than(other)
        raise ValueError('Operator not supported!')
        
    cpdef bint less_than(self, other):
        """Determines if loss is less

        :param other: another score item
        :rtype:ool 
        """
        raise NotImplementedError

    cpdef bint greater_than(self,other):
        """Deteremines if score is higher

        :param other: another score item
        :rtype: bool
        """
        raise NotImplementedError

    cpdef double score(self):
        """Scores the current item

        :rtype: double
        """
        raise NotImplementedError
    
cdef class RankScorer(Scorer):
    """A simple rank scorer that records accuracy at different 
    points in a rank
    """

    def __init__(self,at1,at10,mrr):
        """Initialize a rank score instance 


        :param at1: accuracy at 1
        :param at10: accuracy at 10
        :param mrr: mean reciprocal rank
        """
        self.at1  = at1
        self.at10 = at10
        self.mrr  = mrr

    cpdef double score(self):
        """Sums the three rank scores

        :rtype: double
        """
        cdef double at1  = self.at1
        cdef double at10 = self.at10
        cdef double mrr  = self.mrr

        return (at1+at10+mrr)

    cpdef bint less_than(self, other):
        """Determines if loss is less

        :param other: another score item
        :rtype:ool 
        """
        cdef double score1 = self.score()
        cdef double score2 = other.score()
        return score1 < score2
    
    cpdef bint greater_than(self, other):
        """Determines if loss is less

        :param other: another score item
        :rtype:ool 
        """
        cdef double score1 = self.score()
        cdef double score2 = other.score()
        return score1 > score2

    def __reduce__(self):
        return RankScorer,(self.at1,self.at10,self.mrr)


# ############### cli

def params():
    """Dataset loader options """
    return ({},[])

class BowModel(object):
    pass 

def bow_data(config):
    """build the bag of words data

    :param config: main configuration
    :rtype: None
    """
    dtypes = ["train","test","valid"]
    s,t,sd,td,table,_ = load_aligner_data(config)
    rank_out = os.path.join(config.dir,"ranks.data")
    rlens = []

    for k,d in enumerate(dtypes):
        items = get_rdata(config,sd,td,ttype=d)
        en,gold = items[1]
        en_original = items[-1]
        #en_original = items[-1].tolist()
        ranks = items[0]
        rlens.append(ranks.shape[0])
        dout = os.path.join(config.dir,"%s.data" % d)
        dobj = RankDataset(en,en_original,gold)
        #dobj.dump_dataset(dout)
        dobj.dump(dout)
        if k == 0:
            with open(rank_out,'wb') as my_ranks:
                np.savez(my_ranks,ranks,items[2])
                
    assert len(set(rlens)) == 1,'ranks different size'

    ## make a simple base model object
    basemodel = BowModel()
    basemodel.elen = len(td)
    basemodel.flen = len(sd)
    basemodel.elex = td
    basemodel.flex = sd

    base_out = os.path.join(config.dir,"base.model")
    with open(base_out,'w') as my_model:
        pickle.dump(basemodel,my_model)


def __check_language_file(dtype,atraining):
    lang_file = ''
    if dtype == 'train': lang_file = "%s%s" % (atraining,".language")
    elif dtype == 'valid': lang_file = "%s%s" % (atraining,"_val.language")
    elif dtype == 'test': lang_file = "%s%s" % (atraining,"_test.language")
    if lang_file and os.path.isfile(lang_file):
        return True
    return False

def setup_aligner_data(config):
    """build the dataset objects

    :param config: main zubr configuration object
    :type config: zubr.util.config.ConfigAttrs
    :rtype: None
    """
    dtypes = ["train","test","valid"]
    flex,elex = get_lex(config.align_dir)
    rank_out = os.path.join(config.dir,"ranks.data")
    rlens = []
    
    for k,d in enumerate(dtypes):
        poly = False 
        
        ### check if language file exists
        if __check_language_file(d,config.atraining): poly = True
        items = get_rdata(config,flex,elex,ttype=d,poly=poly)
        en,gold = items[1]
        #en_original = items[-1]
        en_original = items[4]
        ranks = items[0]
        rlens.append(ranks.shape[0])
        dout = os.path.join(config.dir,"%s.data" % d)
        if poly: dobj = RankDataset(en,en_original,gold,langs=items[-1])
        else: dobj = RankDataset(en,en_original,gold)
        #dobj.dump_dataset(dout)
        dobj.dump(dout)
        if k == 0:
            with open(rank_out,'wb') as my_ranks:
                np.savez(my_ranks,ranks,items[2])

    assert len(set(rlens)) == 1,'ranks different size'

def main(config):
    """main execution function, should only be called as
    part of a pipeline

    :param config: main configuration
    """
    if 'aligner' in config.rmodel:
        setup_aligner_data(config)

    elif 'bow' in config.rmodel:
        bow_data(config)
        
    else:
        raise ValueError('unknown dataset type')
