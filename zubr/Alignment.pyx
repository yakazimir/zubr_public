# -*- coding: utf-8 -*-

"""

This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Classes for all alignment models 

"""
import traceback
import logging
import sys
import os
import time
import gzip
import cPickle as pickle
from zubr.util import ConfigAttrs

from zubr.util.alignment_util import (
    print_table,
    load_aligner_data,
    copy_settings,
    output_path,
    build_sparse2d
)
    
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Datastructures cimport Sparse2dArray
from zubr.util.config import ConfigAttrs
from zubr.Alg cimport binary_insert_sort
import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound,cdivision
from zubr.util.config import ConfigAttrs
from libc.stdlib cimport malloc, free

cdef class AlignerBase(ZubrSerializable):

    """A base class for aligners"""

    @classmethod
    def load_data(cls,path='',config=None):
        """Load an aligner with either a configuration object or path to data

        :param path: path to aligner data
        :tyupe path: str
        :param config: an aligner configuration
        :type config: zubr.util.config.ConfigAttrs
        """
        raise NotImplementedError
   
    cpdef void train(self,config=None):
        """Train an alignment model 

        :param config: the aligner configuration 
        :rtype: None 
        """
        raise NotImplementedError

cdef class SequenceAligner(AlignerBase):
    """Sequence to sequence aligners base class"""

    cdef void _train(self,np.ndarray f,np.ndarray e):
        """Main training method for distortion model

        :param f: the foreign side of data 
        :param e: the english side of the data 
        :maxiter: max iterations for initialization model (if used) 
        :maxiter2: the maximum models for main model 
        """
        raise NotImplementedError

    cdef Alignment _align(self,int[:] f,int[:] e):
        """Align an intput output pair 

        :param f: the foreign output
        :param e: the english input 
        :returns: An alignment object 
        """
        raise NotImplementedError

    def backup(self,wdir,tname='table'):
        """Back up the model using numpy as an alternative to pickle 

        :param wdir: the working directory 
        :param tname: the name of the main numpy file 
        """
        raise NotImplementedError
                        
    @classmethod 
    def load_backup(cls,config,tname='table'):
        """Load a numpy based backup and return aligner instance

        :param config: the configuration object
        :param tname: table 
        """
        raise NotImplementedError

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef void _rank(self,int[:] en,np.ndarray rl,int[:] sort):
        """Use aligner to rank input against a rank list of item 

        :param en: the english vector input 
        :param rl: the ranked list
        :param sort: a list to sort
        """
        cdef int rank_size = rl.shape[0]
        cdef int i,end,start,mid
        cdef double prob,sprob
        cdef Alignment alignment
        cdef double[:] problist = np.zeros((rank_size,),dtype='d')

        for i in range(rank_size):
            alignment = self._align(rl[i],en)
            prob = alignment.prob
            problist[i] = prob
            binary_insert_sort(i,prob,problist,sort)


    cpdef void align_dataset(self,np.ndarray foreign,np.ndarray english,out=''):
        """Align a dataset and print giza output to file or stdout

        :param foreign: the foreign dataset 
        :param english: the english dataset
        :param out: the optional file path
        """
        cdef int flen = foreign.shape[0]
        cdef int elen = english.shape[0]
        cdef int i
        cdef Decoding alignment
        out = sys.stdout if out == '' else open(out,"w")

        for i in range(flen):
            for alignment in self._align(foreign[i],english[i]):
                print >>out,alignment

        if out != sys.stdout: out.close()     

    ## properties

    property elex:
        """Access the english lexicon map"""
    
        def __get__(self):
            """Returns the english lexicon map

            :rtype: dict 
            """
            return <dict>(self._elex)

    property flex:
        """Access the foreign lexicon map"""
    
        def __get__(self):
            """Returns the foreign lexicon map

            :rtype: dict 
            """
            return <dict>(self._flex)

    property elen:
        """Size of the english vocabulary"""

        def __get__(self):
            """Returns the size of the english vocabular

            :rtype: int 
            """
            return <int>len(self._elex)

    property flen:
        """Size of the foreign vocabulary"""

        def __get__(self):
            """Returns the size of the foreign vocabular

            :rtype: int 
            """
            return <int>len(self._flex)

    property maxiters:
        """The number of iterations to use for training"""

        def __get__(self):
            """Return the current max iter setting

            :rtype: int
            """
            return <int>self.config.aiters


        def __set__(self,int nmax):
            """Modify the maximum number of iterations 

            :param nmax: the new maximum number of iterations 
            :rtype: None 
            """
            self.config.aiters = nmax

cdef class WordModel(SequenceAligner):

    cpdef void train(self,object config=None):
        """Train the model1 alignemnt 

        :param config: alternative config (possibly empty) 
        """
        cdef object aconfig = config if config else self.config
        cdef bint aligntraining = aconfig.aligntraining

        data = load_aligner_data(aconfig)
        self._train(data[0],data[1])

        ## align the training?
        if aligntraining:
            path = output_path(aconfig)
            self.align_dataset(data[0],data[1])

    def print_table(self,out_file=sys.stdout):
        """Print the lexical translation probabiilties to file 

        :param out_file: where to print the table 
        :type out_file: str 
        :rtype: None 
        """
        print_table(self.table,self._elex,self._flex,out_file)

    cdef void offline_init(self,np.ndarray f,np.ndarray e):
        """Called before running an experiment on a full dataset, passed by default

        :param f: the foreign dataset input 
        :param e: the english dataset input
        """
        pass

    cdef double word_prob(self,int foreign,int english,int identifier=-1):
        """Return translation probability between foreign and input words using the table 

        :param foreign: the foreign word, encoded 
        :param english: the english word, encoded 
        :rtype: double 
        """
        raise NotImplementedError

    cdef double[:,:] model_table(self):
        """This returns word translation datastructure 
        
        motivation : for many downstream uses of these translation 
        models (e.g., graph decoder), it is useful and efficient 
        to have a (non-sparse) 2d array representation of word 
        translation probabilities that supports fast lookup. 
        
        :returns: a typed memory view of the word parameters 
        """
        raise NotImplementedError    


cdef class NonDistortionModel(WordModel):
    """Defines a sequence alignment model without distortion parameters"""

    def __init__(self,np.ndarray table,dict flex,dict elex,object config):
        """Create a non distortion aligner model instance


        :param flex: the foreign lexicon 
        :param elen: the english lexicon 
        :param config: the aligner configuration settings 
        """
        self.table  = table
        self._elex  = elex
        self._flex  = flex
        self.config = config

    @classmethod
    def load_data(cls,path='',config=None):
        """Initialize an aligner by loading data into it and train it 

        -- By specifying only the path to your data, you will load 
        data and aligner with default settings 

        -- By providing a config, you can specify particular settings.

        :param path: the path to the main data
        :type path: str
        :param config: the main configuration (if supplied)
        :type config: zubr.util.config.ConfigAttrs

        """
        settings = config_loader(path,config)
        f,e,sdict,tdict,table,_ = load_aligner_data(settings)
        
        return cls(table,sdict,tdict,settings)

    def backup(self,wdir,tname='ftoe'):
        """Back up the model using numpy as an alternative to pickle 

        :param wdir: the working directory 
        :param tname: the name of the main numpy file 
        """
        ## back up the table
        stime = time.time()

        ## make a model directory
        mdir = os.path.join(wdir,tname)
        if os.path.isdir(mdir):
            self.logger.info('Already backed up! skipping...')
            return
        os.mkdir(mdir)
        fout = os.path.join(mdir,'parameters')
        np.savez_compressed(fout,self.table)

        ## back up the dictionaries
        elex_out = os.path.join(mdir,'e.gz')
        flex_out = os.path.join(mdir,'f.gz')

        with gzip.open(elex_out,'wb') as my_path:
            pickle.dump(self.elex,my_path)
        with gzip.open(flex_out,'wb') as my_path:
            pickle.dump(self.flex,my_path)
        
        ## backup the config
        #cout = os.path.join(mdir,"config.yaml")
        self.config.print_to_yaml(mdir)
        self.logger.info('Backed up %s in %s seconds' % (tname,time.time()-stime))

    @classmethod 
    def load_backup(cls,config,tname='ftoe'):
        """Load a numpy based backup for non distortion model

        :param wdir: the working directory 
        :param tname: the name of the backup and table 
        """
        stime = time.time()
        wdir = config.dir
        mdir = os.path.join(wdir,tname)

        ## load the table 
        afile = os.path.join(mdir,"parameters.npz")
        archive = np.load(afile)
        table = archive["arr_0"]

        ## lexicons 
        elexp = os.path.join(mdir,'e.gz')
        flexp = os.path.join(mdir,'f.gz')
        with gzip.open(elexp,'rb') as el:
            elex = pickle.load(el)
        with gzip.open(flexp,'rb') as fl:
            flex = pickle.load(fl)

        ## configuration
        config = ConfigAttrs()
        config.restore_old(mdir)
        instance = cls(table,flex,elex,config)
        instance.logger.info('Rebuilt %s in %s seconds' % (tname,str(time.time()-stime)))
        return instance

    cdef double word_prob(self,int foreign,int english,int identifier=-1):
        """Computes the following: p( english | foreign) using the table 

        :param foreign: the foreign word 
        :param english: the english word, encoded
        :rtype: float 
        """
        cdef double[:,:] table = self.table
        return table[foreign,english]

    cdef double[:,:] model_table(self):
        """This returns word translation datastructure 
        
        motivation : for many downstream uses of these translation 
        models (e.g., graph decoder), it is useful and efficient 
        to have a (non-sparse) 2d array representation of word 
        translation probabilities that supports fast lookup. 
        
        :returns: a typed memory view of the word parameters 
        """
        cdef double[:,:] table = self.table
        return table
                
cdef class WordDistortionModel(WordModel):

    """Alignment model with distortion parameters defined on word positions"""

    def __init__(self,np.ndarray table,np.ndarray distortion,dict flex,dict elex,object config):
        """Initializes a word distortion model 

        :param table: lexical probability table or parameters
        :param distortion: the distortion parameters
        :param flex: the foreign lexicon 
        :param elex: the english lexicon 
        :param config: the main aligner configuration
        """
        self.table = table
        self.distortion = distortion
        self._flex  = flex
        self._elex  = elex
        self.config = config 

    @classmethod
    def load_data(cls,path='',config=None):
        """Initialize an aligner by loading data into it and train it 

        -- By specifying only the path to your data, you will load 
        data and aligner with default settings 

        -- By providing a config, you can specify particular settings.

        :param path: the path to the main data
        :type path: str
        :param config: the main configuration (if supplied)
        :type config: zubr.util.config.ConfigAttrs

        """
        settings = config_loader(path,config)
        f,e,sdict,tdict,table,distortion = load_aligner_data(settings)
        initialize_dist(distortion,settings.amax)
        
        return cls(table,distortion,sdict,tdict,settings)

    def backup(self,wdir,tname='ftoe'):
        """Back up the model using numpy as an alternative to pickle 

        :param wdir: the working directory 
        :param tname: the name of the main numpy file 
        """
        ## back up the table
        stime = time.time()

        ## make a model directory
        mdir = os.path.join(wdir,tname)
        if os.path.isdir(mdir):
            self.logger.info('Already backed up! skipping...')
            return
        os.mkdir(mdir)
        fout = os.path.join(mdir,'table')
        np.savez_compressed(fout,self.table,self.distortion)

        ## back up the dictionaries
        elex_out = os.path.join(mdir,'e.gz')
        flex_out = os.path.join(mdir,'f.gz')

        with gzip.open(elex_out,'wb') as my_path:
            pickle.dump(self.elex,my_path)
        with gzip.open(flex_out,'wb') as my_path:
            pickle.dump(self.flex,my_path)
        
        ## backup the config
        #cout = os.path.join(mdir,"config.yaml")
        self.config.print_to_yaml(mdir)
        self.logger.info('Backed up in %s seconds' % (time.time()-stime))

    @classmethod 
    def load_backup(cls,config,tname='ftoe'):
        """Load a numpy based backup for non distortion model

        :param config: the main configuration
        :param tname: the name of the backup and table 
        """
        stime = time.time()
        wdir = config.dir 
        mdir = os.path.join(wdir,tname)

        ## load the table 
        afile = os.path.join(mdir,"parameters.npz")
        archive = np.load(afile)
        table = archive["arr_0"]
        distortion = archive["arr_1"]

        ## lexicons 
        elexp = os.path.join(mdir,'e.gz')
        flexp = os.path.join(mdir,'f.gz')
        with gzip.open(elexp,'rb') as el:
            elex = pickle.load(el)
        with gzip.open(flexp,'rb') as fl:
            flex = pickle.load(fl)

        ## configuration
        config = ConfigAttrs()
        config.restore_old(mdir)
        instance = cls(table,distortion,flex,elex,config)
        instance.logger.info('Rebuilt in %s seconds' % str(time.time()-stime))
        
        return instance         

    cdef double word_prob(self,int foreign,int english,int identifier=-1):
        """Computes the following: p( english | foreign) using the table 

        :param foreign: the foreign word 
        :param english: the english word, encoded
        :rtype: float 
        """
        cdef double[:,:] table = self.table
        return table[foreign,english]

    cdef double[:,:] model_table(self):
        """This returns word translation datastructure 
        
        motivation : for many downstream uses of these translation 
        models (e.g., graph decoder), it is useful and efficient 
        to have a (non-sparse) 2d array representation of word 
        translation probabilities that supports fast lookup. 
        
        :returns: a typed memory view of the word parameters 
        """
        cdef double[:,:] table = self.table
        return table
    
cdef class TreeDistortionModel(TreeModel):
    """Alignment model with distortion parameters defined on target-side tree positions"""
    pass

### PARTICULAR IMPLEMENTATIONS

cdef class IBMM1(NonDistortionModel):

    """Implementation of the IBM Model 1 aligner 


    Here's an example invocation of an aligner, through training:

    >>> from zubr.Alignment import IBMM1 
    >>> a = IBMM1.load_data(path='examples/aligner/hansards') 
    >>> a.train()
    """

    cdef void _train(self,np.ndarray f,np.ndarray e):
        """Main training method for distortion model

        :param f: the foreign side of data 
        :param e: the english side of the data 
        :maxiter: max iterations for initialization model (if used) 
        :maxiter2: the maximum models for main model 
        """
        cdef object config = self.config
        cdef int maxiters = config.aiters
        cdef double[:,:] table = self.table
        cdef int elen = self.elen,flen = self.flen

        self.logger.info('Training the main model...')
        
        learn_model1(table,elen,flen,e,f,maxiters,self.logger)

    cdef Alignment _align(self,int[:] f,int[:] e):
        """Align an intput output pair using lexical translation probabilities 

        :param f: the foreign output
        :param e: the english input 
        :returns: An alignment object 
        """
        cdef double[:,:] table = self.table
        return _decode_m1(f,e,table)

    def __reduce__(self):
        ## pickle implementation
        return IBMM1,(self.table,self.flex,self.elex,self.config)


cdef class IBMM2(WordDistortionModel):

    """Implementation of the IBM Model 2 aligner 


    Here's an example invocation of an aligner, through training:

    >>> from zubr.Alignment import IBMM1 
    >>> a = IBMM1.load_data(path='examples/aligner/hansards') 
    >>> a.train()

    """

    cdef void _train(self,np.ndarray f,np.ndarray e):
        """Main training method for distortion model

        :param f: the foreign side of data 
        :param e: the english side of the data 
        :maxiter: max iterations for initialization model (if used) 
        :maxiter2: the maximum models for main model 
        """
        cdef double[:,:] table = self.table
        cdef double[:,:,:,:] distortion = self.distortion
        cdef int elen = self.elen, flen = self.flen
        cdef object config = self.config
        cdef int maxiter = config.aiters
        cdef int maxiter2 = config.aiters2
        cdef int maxl = config.amax

        self.logger.info('Initializing model....')
        learn_model1(table,elen,flen,e,f,maxiter,self.logger)
        
        self.logger.info('Now retraining the main model')
        learn_model2(table,distortion,elen,flen,e,f,maxiter2,maxl,self.logger)

    cdef Alignment _align(self,int[:] f,int[:] e):
        """Align an input/output pair using IBM Model2

        :param f: the foreign output
        :param e: the english input 
        :returns: An alignment object 
        """
        cdef double[:,:] table = self.table
        cdef double[:,:,:,:] distortion = self.distortion
        
        return _decode_m2(f,e,table,distortion)

    def __reduce__(self):
        ## pickle implementation
        return IBMM2,(self.table,self.distortion,self.flex,self.elex,self.config)


cdef class TreeModel(SequenceAligner):
    """Implementation of tree model aligner (a variant of Model 2) """
    pass

cdef class TreeModel2(TreeDistortionModel):
    """Implementation of tree model aligner (a variant of Model 2) """
    pass

### SPARSE MODELS

## sparse representation

cdef class Sparse2dModel(Sparse2dArray):
    """A sparse representation of lexical translation parameters"""
    
    @classmethod
    def from_data(cls,edata,fdata,flex):
        """Create a sparse lexical model instance from data (e.g., training data)

        :param edata: the english data 
        :param fdata: the foreign data
        :param flex: the foreign lexicon 
        """
        ## build data
        dim1,dim2,span = build_sparse2d(edata,fdata,flex)
        return cls(dim1,dim2,span)

cdef class SparseWordModel(WordModel):
    """A word sequence model that uses a sparse representation 
    for the translation parameters
    """

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef double[:,:] model_table(self):
        """Makes a non-sparse representation of word pair parameters. 
        
        motivation : for many downstream uses of these translation 
        models (e.g., graph decoder), it is useful and efficient 
        to have a (non-sparse) 2d array representation of word 
        translation probabilities that supports fast lookup. 
        
        :returns: a typed memory view of the word parameters 
        """
        cdef double[:] sparse_table = self.table
        cdef Sparse2dModel lookup = self.sparse2d
        cdef int[:,:] spans = lookup.spans
        cdef int[:] dim2 = lookup.dim2
        cdef int elen = self.elen, flen = self.flen
        cdef double[:,:] table = np.zeros((flen,elen),dtype='d')
        cdef int i,k,evalue
        
        for i in range(flen):
            for k in range(spans[i][0],spans[i][1]):
                evalue = dim2[k]
                table[i,evalue] = sparse_table[k]

        return table 
    
cdef class SparseNonDistortionModel(SparseWordModel):

    def __init__(self,table,lookup,flex,elex,config):
        """Create a sparse non distortion alingment model instance

        :param table: the parameter table 
        :param lookup: a sparse 2d array for lookup 
        :param flex: the foreign lexicon
        :param elex: the english lexicon
        :param config: the aligner configuration 
        """
        self.table    = table
        self._flex    = flex
        self._elex    = elex
        self.config   = config
        self.sparse2d = lookup

        ## log the parameter size 
        self.logger.info('Loaded with %d parameters (out of full %d)....' %\
                             (self.table.shape[0],len(self._elex)*len(self._flex)))

    @classmethod
    def load_data(cls,path='',config=None):
        """Initilalize an aligner by loading data into it

        -- By specifying only the path, you will load your
        data and aligner with default settings

        -- With a configuration you can provide custom 
        settings 

        :param path: the path to main data 
        :param config: the optional configuration
        """
        settings = config_loader(path,config)
        f,e,sdict,tdict,init,_ = load_aligner_data(settings)
        
        ## load sparse array
        lookup = Sparse2dModel.from_data(e,f,sdict)
        psize = lookup.size
        table = np.ndarray((psize,),dtype='d')
        table.fill(init)
                
        return cls(table,lookup,sdict,tdict,settings)

    def backup(self,wdir,tname='ftoe'):
        """Backup this model for later use in Pipeline 

        Note : this model automatically backs up the sparse2d
        array, rather than calling a special method in the
        array instance 

        :param wdir: the working directory 
        """
        cdef Sparse2dModel lookup = self.sparse2d
        
        stime = time.time()
        mdir = os.path.join(wdir,tname)
        if os.path.isdir(mdir):
            self.logger.info('Already backed up! skipping...')
            return
        os.mkdir(mdir)
        fout = os.path.join(mdir,'sparse_parameters')
        np.savez_compressed(fout,self.table)

        ## backup lookup model
        sout = os.path.join(mdir,'sparse_components')
        np.savez_compressed(sout,lookup.dim1,lookup.dim2,lookup.spans)

        ## back up the dictionaries
        elex_out = os.path.join(mdir,'e.gz')
        flex_out = os.path.join(mdir,'f.gz')

        with gzip.open(elex_out,'wb') as my_path:
            pickle.dump(self.elex,my_path)
        with gzip.open(flex_out,'wb') as my_path:
            pickle.dump(self.flex,my_path)

        self.config.print_to_yaml(mdir)
        self.logger.info('Backed up %s in %s seconds' % (tname,time.time()-stime))

    @classmethod 
    def load_backup(cls,config,tname='ftoe'):
        """Load a model from a backup 

        :param config: the aligner configuration 
        """
        stime = time.time()
        wdir = config.dir
        mdir = os.path.join(wdir,tname)

        ## parameters 
        afile = os.path.join(mdir,"sparse_parameters.npz")
        archive = np.load(afile)
        table = archive["arr_0"]

        ## sparsed2d instance
        sfile = os.path.join(mdir,"sparse_components.npz")
        sarchive = np.load(sfile)
        dim1 = sarchive["arr_0"]
        dim2 = sarchive["arr_1"]
        spans = sarchive["arr_2"]
        lookup = Sparse2dModel(dim1,dim2,spans)

        ## lexicons 
        elexp = os.path.join(mdir,'e.gz')
        flexp = os.path.join(mdir,'f.gz')
        with gzip.open(elexp,'rb') as el:
            elex = pickle.load(el)
        with gzip.open(flexp,'rb') as fl:
            flex = pickle.load(fl)

        config = ConfigAttrs()
        config.restore_old(mdir)

        return cls(table,lookup,flex,elex,config)


cdef class SparseDistortionModel(SparseWordModel):
    pass

cdef class SparseIBMM1(SparseNonDistortionModel):

    cdef void _train(self,np.ndarray f,np.ndarray e):
        """Main training method for distortion model

        :param f: the foreign side of data 
        :param e: the english side of the data 
        :maxiter: max iterations for initialization model (if used) 
        :maxiter2: the maximum models for main model 
        """
        cdef object config = self.config
        cdef int maxiters = config.aiters
        cdef double[:] table = self.table
        cdef Sparse2dModel lookup = self.sparse2d 
        cdef int elen = self.elen,flen = self.flen
        
        self.logger.info('Training the main model...')
        learn_model1_sparse(table,lookup,elen,flen,e,f,maxiters,self.logger)

    cdef Alignment _align(self,int[:] f,int[:] e):
        """Align an intput output pair using lexical translation probabilities 

        :param f: the foreign output
        :param e: the english input 
        :returns: An alignment object 
        """
        cdef double[:] table = self.table
        cdef Sparse2dModel lookup = self.sparse2d
        return _decode_m1_sparse(f,e,table,lookup)

    cdef double word_prob(self,int foreign,int english,int identifier=-1):
        """Computes the following: p( english | foreign) using the table 

        :param foreign: the foreign word 
        :param english: the english word, encoded
        :rtype: float 
        """
        cdef double[:] table = self.table
        cdef Sparse2dModel lookup = self.sparse2d
        cdef int pair_id = lookup.find_id(foreign,english)
        
        if pair_id == -1: return 0.0
        return table[pair_id]


cdef class SparseIBMM2(SparseDistortionModel):
    pass 

## alignment objects

cdef class Alignment:
    """an alignment object for representing decoder output

    An example invocation: 

    >>> from zubr.Alignment import Alignment 
    >>> a = Alignment.make_empty(4,5)
    >>> a.prob 
    0.0 

    """
    
    def __init__(self,slen,tlen,prob,problist,ml):
        """Initialize a alignment storage instance 

        :param slen: the source (or foreign) len
        :type slen: int
        :param tlen: the target (or english) len
        :type tlen: int
        :param prob: the overall alignment probability 
        :type prb: float 
        :param ml: the most likely alignment positions 
        :type ml: np.ndarray 
        """
        self.slen = slen
        self.tlen = tlen
        self.prob = prob
        self.problist = problist
        self.ml = ml
        
    def __iter__(self):
        return iter(self.alignments())

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef list _find_best(self):
        """find the viterbi alignment via greedy search

        -- assumes that 0 is None 

        :returns: Decoding object with alignment positions
        :rtype: list
        """
        cdef int[:] bestl = self.ml
        cdef double[:,:] problist = self.problist
        cdef int tlen = self.tlen
        cdef int indx
        cdef Decoding decoding = Decoding(tlen)
        cdef int[:] ts_array = decoding.positions
        
        for i in range(tlen):
            indx = bestl[i]
            decoding.prob *= problist[i][indx]
            if indx <= 0: continue
            ts_array[i] = indx

        return [decoding]

    def alignments(self,k=1):
        """main python method for finding alignments

        :param k: number of alignments to return
        :type k: int
        :returns: list of alignments
        :rtype: list
        """
        return self._find_best()

    @classmethod
    def make_empty(cls,slen,tlen):
        """Initialize an empty alignment object

        :param slen: source input length
        :param tlen: target input length
        :returns: empty alignment instance
        :rtype: Alignment
        """
        ml = np.zeros(tlen,dtype=np.int32)
        problist = np.zeros((tlen,slen),dtype='d')
        return cls(slen,tlen,0.0,problist,ml)


## decoding item

cdef class Decoding:
    """A representation of a viterbi alignment decoding 

    An example viterbi alignment: 

    >>> from zubr.Alignment import Decoding
    >>> decoding = Decoding(5) 
    >>> pvector = decoding.positions 
    >>> pvector[0] = 2
    >>> pvector[3] = 1
    >>> print decoding
    0-1 3-0
    """

    def __init__(self,tlen):
        """initializes an empty decoding instance 

        :param tlen: the length of the target (or e) sequence
        """
        self.tlen = tlen
        self.positions = np.zeros(tlen,dtype=np.int32)
        self.prob = 1.0

    cpdef unicode giza(self):
        """Prints giza format given a viterbi alignment

        -- assumes that source 0 is None, which does
        not get printed in the giza string (and everything
        else gets decremented by one) 

        :returns: giza formatted string
        :rtype: unicode
        """
        cdef int[:] alignment = self.positions
        cdef int tlen = self.tlen
        return generate_giza(tlen,alignment)

    def __str__(self):
        ## string method implementation
        return self.giza()

## helper functions

def config_loader(path,config):
    """Returns a configuration object depending on input 

    :param path: an optional path pointing to data 
    :type path: str
    :param config: a (potentially empty) existing configuration 
    :type config: zubr.util.config.ConfigAttrs or None
    :returns: an aligner configuration 
    """
    if config == None:
        parser = argparser()
        new_config = parser.get_default_values()
        new_config.atraining = path
        settings = new_config
    else:
        settings = ConfigAttrs()
        copy_settings(config,settings)

    return settings

## c methods

## generate giza format

@boundscheck(False)
@wraparound(False)
cdef unicode generate_giza(int tlen,int[:] positions):
    """Generate a giza formatted string from a decoder input

    :param tlen: the target string length 
    :param positions: the vitebri alignment positions
    :returns: unicode string representation of alignment in giza format
    """
    cdef unicode finalstr = u''
    cdef int i,sval

    for i in range(tlen):
        sval = positions[i]
        if sval <= 0: continue
        sval -= 1
        finalstr += u"%d-%d " % (i,sval)

    return finalstr


## decode with model 1

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef Alignment _decode_m1(int[:] f,int[:] e,double[:,:] table):
    """Decode with model1/lexical translation probability parameters

    :param f: the foreign side encoding
    :param e: the english side encoding
    :param table: the model1 parameters
    """
    cdef int slen = f.shape[0]
    cdef int tlen = e.shape[0]
    cdef Alignment record = Alignment.make_empty(slen,tlen)
    cdef double[:,:] problist = record.problist
    cdef int[:] ml = record.ml
    cdef int k,i
    cdef double z,overallprob = 0.0
    cdef double score
    cdef double div = float(slen)**float(tlen)

    for k in range(tlen):

        ## skip unknown words on e side 
        if e[k] == -1: continue
        z = 0.0
        
        for i in range(slen):
            ## skip over unknown on f side
            if f[i] == -1: continue 
                
            score = table[f[i],e[k]]
            z += score
            problist[k][i] = score
            if score >= problist[k][ml[k]]:
                ml[k] = i

        if overallprob == 0.0: overallprob = z
        else: overallprob *= z

    #normalize for length
    overallprob = overallprob/div
    record.prob = overallprob
    return record

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef Alignment _decode_m1_sparse(int[:] f,int[:] e,double[:] table,Sparse2dModel lookup):
    """Decode IBM model 1 for sparse parameter representation

    :param f: the foreign input 
    :param e: the english input  
    :param table: the translation parameters 
    :param lookup: a sparse 2d-array for looking up pair identifiers
    """
    cdef int slen = f.shape[0]
    cdef int tlen = e.shape[0]
    cdef Alignment record = Alignment.make_empty(slen,tlen)
    cdef double[:,:] problist = record.problist
    cdef int[:] ml = record.ml
    cdef int k,i
    cdef double z,overallprob = 0.0
    cdef double score
    cdef double div = float(slen)**float(tlen)
    cdef int pair_id 

    for k in range(tlen):
        if e[k] == -1: continue
        z = 0.0
        
        for i in range(slen):
            if f[i] == -1: continue
            pair_id = lookup.find_id(f[i],e[k])
            ## find the pair id
            if pair_id == -1: continue
            score = table[pair_id]
            z += score
            problist[k][i] = score
            if score >= problist[k][ml[k]]: ml[k] = i

        if overallprob == 0.0: overallprob = z
        else: overallprob *= z

    overallprob = overallprob/div
    record.prob = overallprob
    return record
                
            
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef Alignment _decode_m2(int[:] f,int[:] e,double[:,:] table,double[:,:,:,:] distortion):
    """Decoder for IBM Model2 with distortion parameters

    :param f: the foreign side 
    :param e: the english side 
    :param table: the model1 parameters 
    :param distortion: the distortion parameters
    """
    cdef int slen = f.shape[0]
    cdef int tlen = e.shape[0]
    cdef Alignment record = Alignment.make_empty(slen,tlen)
    cdef double[:,:] problist = record.problist
    cdef int[:] ml = record.ml
    cdef int k,j
    cdef double z,overallprob = 0.0
    cdef double score

    for k in range(tlen):
        if e[k] == -1: continue
        z = 0.0

        for j in range(slen):
                
            if f[j] == -1: continue
            score = (table[f[j],e[k]])*(distortion[slen-1,tlen-1,k,j])
            z += score
            problist[k][j] = score
            if score >= problist[k][ml[k]]:
                ml[k] = j

        if overallprob == 0.0: overallprob = z
        else: overallprob *= z

    record.prob = overallprob
    return record
    

## TRAIN MODEL2 PARAMETERS

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void learn_model2(double[:,:] table, ## probability table
                          double[:,:,:,:] distortion, ## distortion parameters
                          int elen,int flen,
                          np.ndarray target,np.ndarray source,
                          int maxiter,
                          int maxl,
                          object logger,
                          ):
    """Learn model 2 aligner with distortion parameters 

    
    :param table: the lexical translation parameters 
    :param distortion: the distortion parameters 

    :param edata: english training data 
    :param fdata: the foreign training data 

    :param maxiter: the maximum iterations (for model2!) 

    :param logger: instance logger for recording likelihood, etc.. 
    :rtype: None 
    """
    cdef int comparisons, size = target.shape[0]
    cdef int slen,tlen,k,j,e,f,i,K,I
    cdef int iterations = 0
    cdef int[:] sourcet
    cdef int[:] targett
    cdef double logprob = 1e-1000
    cdef double newlogprob,count,score,news
    cdef double total,paircount,z
    cdef double[:,:,:,:] dcounter
    cdef double[:,:,:] gcounts2
    cdef double[:,:] ccounter
    cdef double[:] gcounts1

    stime = time.time()
    ccounter = np.zeros((flen,elen),dtype='d')
    gcounts1 = np.zeros((flen,),dtype='d')
    dcounter = np.zeros((maxl,maxl,maxl,maxl),dtype='d')
    gcounts2 = np.zeros((maxl,maxl,maxl),dtype='d')

    while True:

        iterations += 1
        if iterations > maxiter: break
        if iterations > 1: logprob = newlogprob
        newlogprob = 1e-1000

        ## reset counters
        ccounter[:,:] = 0.0
        gcounts1[:] = 0.0
        dcounter[:,:,:,:] = 0.0
        gcounts2[:,:,:] = 0.0            

        # e-step 
        for i in range(size):
            sourcet = source[i]
            targett = target[i]
            slen = sourcet.shape[0]
            tlen = targett.shape[0]

            for k in range(tlen):
                e = targett[k]
                z = 0.0
                for j in range(slen):
                    z += (table[sourcet[j],e])*(distortion[slen-1,tlen-1,k,j])

                for j in range(slen):
                    f = sourcet[j]
                    count = (distortion[slen-1,tlen-1,k,j]*table[f,e])/z
                    ccounter[f,e] += count
                    gcounts1[f] += count
                    dcounter[slen-1,tlen-1,k,j] += count
                    gcounts2[slen-1,tlen-1,k] += count
                        
                newlogprob += log(z)
                    
        # m-step
                
        ## co-occurence updates
        for i in range(flen):
            total = gcounts1[i]

            ## ignore unobserved items
            if total == 0.0: continue
                
            for k in range(elen):
                paircount = ccounter[i,k]
                if paircount == 0.0:
                    table[i,k] = 0.0
                else:
                    table[i,k] = (paircount/total)

        ## distortion updates
        for K in range(maxl):
            for I in range(maxl):
                for k in range(K+1):
                    total = gcounts2[K,I,k]

                    ## ignore unobserved items
                    if total == 0.0: continue 

                    total += 0.0
                    for j in range(I+1):
                        paircount = dcounter[K,I,k,j]
                        if paircount == 0.0:
                            distortion[K,I,k,j] = 0.0
                        else:
                            distortion[K,I,k,j] = (paircount/total)
                            
        ## stop early if likelihood decreases
        if iterations > 2 and logprob > newlogprob: break
        logger.info("Iteration: %d,\t log likelihood: %f" % (iterations,newlogprob))

    ## finish and log the time 
    logger.info('Finished training Model1 in %f seconds' % (time.time()-stime))
    

## TRAIN MODEL1 PARAMETERS

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void learn_model1_sparse(double[:] table,
                                  Sparse2dModel lookup,
                                  int elen,int flen,
                                  np.ndarray edata,np.ndarray fdata,
                                  int maxiter,
                                  object logger,
        ):
    """ Learn sparse model 1 parameters from a dataset

    :param table: the (sparse) translation model parameters) 
    :param lookup: the sparse word representation 
    :param elen: the length of the english vocabulary (do we need this?)
    :param flen: the length of the foreign vocabular
    :param logger: instance logger 
    """
    cdef int size = edata.shape[0]
    cdef int iterations = 0

    ## counts
    cdef double[:] globalcounts
    cdef double[:] counter
    cdef double total,paircount

    ## 
    cdef double logprob = 1e-1000,newlogprob,z
    cdef int i,k,e,f,j

    ##
    cdef int[:] dim2 = lookup.dim2
    cdef int[:,:] spans = lookup.spans

    ## parallel data instacnes
    cdef int[:] sourcet
    cdef int[:] targett
    cdef int slen,tlen
    cdef int pair_id
    cdef int num_pairs = sum([w.shape[0]*edata[k].shape[0] for k,w in enumerate(fdata)])
    cdef int current_pair

    ## pair ids
    cdef int *ids = <int *>malloc(num_pairs*sizeof(int))
    
    ## time
    stime = time.time()

    ## counters
    globalcounts = np.zeros((flen,),dtype='d')
    counter = np.zeros((dim2.shape[0]),dtype='d')

    ## find pair ids

    try:
        
        ## find the id pairs using the lookup (is expensive to do many times)
        current_pair = 0
            
        for i in range(size):
            sourcet = fdata[i]
            targett = edata[i]
            slen = sourcet.shape[0]
            tlen = targett.shape[0]

            for k in range(tlen):
                e = targett[k]

                for j in range(slen):
                    f = sourcet[j] 
                    pair_id = lookup.find_id(f,e)
                    ids[current_pair] = pair_id
                    current_pair += 1
                    
        ## do the training

        while True:
            iterations += 1

            if iterations > maxiter: break 
            if iterations > 1: logprob = newlogprob

            ## log likelihood counter 
            newlogprob = 1e-1000

            ## reset the counters to zero 
            counter[:] = 0.0
            globalcounts[:] = 0.0
            current_pair = 0

            ## iterate through the dataset : E-step 
            for i in range(size):

                ## instance vector 
                sourcet = fdata[i]
                targett = edata[i]
                slen = sourcet.shape[0]
                tlen = targett.shape[0]

                for k in range(tlen):
                    z = 0.0

                    for j in range(slen):
                        pair_id = ids[current_pair+j]
                        z += table[pair_id]

                    for j in range(slen):
                        pair_id = ids[current_pair]
                        count = table[pair_id]/z
                        globalcounts[sourcet[j]] += count
                        counter[pair_id] += count
                        current_pair += 1

                    newlogprob += log(z)

            ### M-step

            for i in range(flen):
                total = globalcounts[i]
                if total == 0.0: continue

                ## go through adjacency
                for k in range(spans[i][0],spans[i][1]):
                    paircount = counter[k]
                    if paircount == 0.0: table[k] = 0.0
                    else: table[k] = paircount/total

            ## stop early if log probability is bigger than before
            if iterations > 2 and logprob > newlogprob: break
            logger.info('Iteration: %d,\t log likelihood: %f' % (iterations,newlogprob))
                                                
        ## log the time 
        logger.info('Finished training Model1 in %f seconds' % (time.time()-stime))
        
    finally:
        free(ids)

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void learn_model1(double[:,:] table, ## probability table m
                        int elen,int flen,## size of the two vocabularies 
                        np.ndarray edata,np.ndarray fdata, ## datasets 
                        int maxiter, ## maximum iterations
                        object logger, ## instance logger 
                        ):
    """learn lexical translation probabilities using expectation maximization algorithm 

    :param table: the translation probability table 

    :param elen: the size of the english vocabulary 
    :param flen: the size of the foreign vocabulary 

    :param edata: the english portion of the parallel data 
    :param fdata: the foreign portion of the parallel data 

    :param maxiter: the maximum number of iterations 
    :param logger: the logger object
    """
    cdef int size = edata.shape[0]
    cdef int iterations = 0
    
    cdef double[:,:] counter
    cdef double[:] globalcounts
    cdef double total,paircount
    
    cdef double logprob = 1e-1000,newlogprob,z
    cdef int i,k,e,f,j

    ## parallel data instacnes
    cdef int[:] sourcet
    cdef int[:] targett
    cdef int slen,tlen

    ## time
    stime = time.time()

    ## counters 
    counter      = np.zeros((flen,elen),dtype='d')
    globalcounts = np.zeros((flen,),dtype='d')

    while True:
        iterations += 1

        ## reached maximum?
        if iterations > maxiter: break
        if iterations > 1: logprob = newlogprob

        ## log likelihood counter 
        newlogprob = 1e-1000

        ## reset the counters to zero 
        counter[:,:] = 0.0
        globalcounts[:] = 0.0

        ## iteration through the data, E-STEP 
        for i in range(size):

            ## instance vector 
            sourcet = fdata[i]
            targett = edata[i]
            slen = sourcet.shape[0]
            tlen = targett.shape[0]

            for k in range(tlen):
                e = targett[k] 
                z = 0.0
                
                for j in range(slen):
                    z += table[sourcet[j],e]
                for j in range(slen):
                    f = sourcet[j]
                    count = table[f,e]/z
                    counter[f,e] += count
                    globalcounts[f] += count

                newlogprob += log(z)

        ## update using counts, M-STEP

        for i in range(flen):
            total = globalcounts[i]
            if total == 0.0: continue

            for k in range(elen):
                paircount = counter[i,k]
                if paircount == 0.0: table[i,k] = 0.0
                else: table[i,k] = paircount/total

        ## stop early if log probability is bigger than before
        if iterations > 2 and logprob > newlogprob: break
        logger.info('Iteration: %d,\t log likelihood: %f' % (iterations,newlogprob))
        
    ## finish and log the time 
    logger.info('Finished training Model1 in %f seconds' % (time.time()-stime))

## aligner factory

ALIGNERS = {
    "ibm1"        : IBMM1,
    "ibm2"        : IBMM2,
    "treemodel"   : None,
    "sparse_ibm1" : SparseIBMM1,
}

def Aligner(model_type):
    """Factor method for returning an aligner type

    :param model_type: the type of model
    :type model_type: basestring
    """
    mtype = model_type.lower()
    if mtype not in ALIGNERS:
        raise ValueError('Unknown aligner type: %s' % mtype)
    return ALIGNERS[mtype]

## CLI

def params():
    """main parameters for running the aligners and/or aligner experiments

    :rtype: tuple
    :returns: description of option types with name, list of options 
    """

    options = [
        ("--amax","amax",90,int,
         "maximum sentence length [default=90]","GeneralAligner"),
        ("--fsuffix","source",'f',"str",
         "source file extension [default='f']","GeneralAligner"),
        ("--esuffix","target",'e',"str",
         "target file extension [default='e']","GeneralAligner"),
        ("--modeltype","modeltype",'ibm1',"str",
         "type of alignment model [default='ibm1']","GeneralAligner"),
        ("--aiters","aiters",5,int,
         "maximum iterations for aligner [default=10]","GeneralAligner"),
        ("--aiters2","aiters2",5,int,
         "maximum iterations for aligner model2 [default=10]","GeneralAligner"),
        ("--aiters3","aiters3",5,int,
         "maximum iterations for tree model [default=5]","TreeAligner"),
        ("--atraining","atraining","","str",
         "location of aligner training data [default='']","GeneralAligner"),
        ("--aligntraining","aligntraining",False,
         "bool","align training data [default=False]","GeneralAligner"),
        ("--aligntesting","aligntesting",False,
         "bool","align testing data [default=False]","GeneralAligner"),
        ("--emax","emax",100,int,
         "maximum size of e side input [default=100]","GeneralAligner"),         
        ("--trainstop","trainstop",10000000,int,
         "for models > 2, place where core training data stops [default=1000000]","TreeAligner"),
        ("--maxtree","maxtre",10,int,
         "for tree model, largest tree size [default=10]","TreeAligner"),
        ("--amode","amode","train","str",
            "aligner mode [default='train']","GeneralAligner"),
        ("--lower","lower",True,"bool",
            "map words to lowercase globally [default=True]","GeneralAligner"),
        ("--encoding","encoding",'utf-8',"str",
            "the aligner encoding [default='utf-8']","GeneralAligner"),   
    ]

    aligner_group = {
        "GeneralAligner" :"General settings for aligners",
        "TreeAligner"    : "Settings for tree aligner (if used)",
    }
    return (aligner_group,options)

def argparser():
    """Return an aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr alignment [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

def main(argv):
    """The main execution function for running aligners 

    :param argv: a configuration object or raw CLI input
    :rtype: None
    """
    
    ## set up the configuration 
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])
        logging.basicConfig(level=logging.DEBUG)

    ## try loading and doing something with aligner
    try: 
        
        aclass = Aligner(config.modeltype)
        ainstance = aclass.load_data(config=config)

        if config.amode == 'train': 
            ainstance.train()

            ## playing aroud with backing up/reloading
            #ainstance.backup(config.dir)
            #reloaded = aclass.load_backup(config)

        elif config.amode == 'test':
            pass 
            
    except Exception,e:
        try: 
            ainstance.logger.error(e,exc_info=True)
        except:
            pass
        traceback.print_exc(file=sys.stdout)

    finally:
        if config.dump_models:
            model_out = os.path.join(config.dir,"base.model")
            ainstance.dump(model_out)

if __name__ == "__main__":
    main(sys.argv[1:])

    
