# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""

from cython cimport boundscheck,wraparound,cdivision
from zubr.Alignment cimport WordModel,TreeModel
from zubr.Alignment import Aligner
#from zubr.NeuralModels import NeuralLearner
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Dataset cimport RankStorage
import os
import sys
import traceback
import numpy as np
cimport numpy as np
import time
import logging 
from zubr.util import ConfigObj,ConfigAttrs
from zubr.util.aligner_util import get_rdata
from zubr.util.termbaseline import run as run_baseline
from zubr.util.polyglot_util import swap_results

DECODERS = {
    "single"   : MonoRankDecoder,
    "polyglot" : PolyglotRankDecoder,
    #"neural"   : NeuralSingleDecoder,
}

cdef class RankerBase(ZubrSerializable):
    """Base class for ranker items"""

    cpdef int rank(self,object config) except -1:
        """Python c/method for ranking a list

        :param config: the ranker configuration 
        :rtype: None 
        """
        raise NotImplementedError

    @classmethod
    def load_model(cls,config):
        """Load instance from configuration 
        
        :param configuration: the main configuration 
        """
        raise NotImplementedError
    
    def train(self,config=None):
        """Train the underlying model 

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        :rtype: None
        """
        raise NotImplementedError

cdef class RankDecoder(RankerBase):
    """Base class for rank decoders"""

    def __init__(self,aligner):
        """Create a rank decoder instance

        :param aligner: the target aligner 
        :type aligner: WordModel 
        """
        self.aligner = <WordModel>aligner

        if isinstance(aligner,TreeModel):
            raise ValueError('Tree models not supported!')

    @classmethod
    def load_model(cls,config):
        """Create a decoder instance with aligner

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        :rtype: RankDecoder
        """
        mtype = Aligner(config.modeltype)
        return cls(mtype.load_data(config=config))

    def train(self,config=None):
        """Train the underlying alignment model 

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        :rtype: None
        """
        self.logger.info('Training the underlying model...')
        self.aligner.train()

    cpdef int rank(self,object config) except -1:
        """Python c/method for ranking a list

        :param config: the ranker configuration 
        :rtype: None 
        """
        raise NotImplementedError

    def __reduce__(self):
        ## general pickle implementation 
        return type(self),(self.aligner,)


cdef class MonoRankDecoder(RankDecoder):
    
    """Rank decoder for multiple languages


    An example invocation: 

    >>> from zubr.RankDecoder import MonoRankDecoder  
    >>> decoder = MonoRankDecoder.load_aligner(path='examples/aligner/hansards')

    """

    cpdef int rank(self,object config) except -1:
        """Rank decode for a multiple datasets/output languages

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        """
        cdef np.ndarray en,rl,freq,order
        cdef np.ndarray[dtype=np.int32_t,ndim=1] gid
        cdef WordModel aligner = <WordModel>self.aligner
        cdef dict flex = aligner.flex
        cdef dict elex = aligner.elex

        self.logger.info('Evaluating to %s set' % config.eval_set)
        rl,inp,order,freq,enorig = get_rdata(config,flex,elex,ttype=config.eval_set)
        en,gid = inp

        ## main decode function
        _decode_single(rl,en,aligner,gid,config.dir,self.logger)

cdef class PolyglotRankDecoder(RankDecoder):
    """Decoder or rank with datasets that have multiple output languages"""

    cpdef int rank(self,object config) except -1:
        """Rank decode for a multiple datasets/output languages

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        """
        cdef WordModel aligner = <WordModel>self.aligner
        cdef np.ndarray en,rl,freq,order
        cdef np.ndarray[dtype=np.int32_t,ndim=1] gid
        cdef dict flex = aligner.flex
        cdef dict elex = aligner.elex
        cdef dict langs = config.lang_ids
        cdef str name

        self.logger.info('Evaluating to %s set' % config.eval_set)
        ## go through different datasets
        for (name,_) in langs.iteritems():
            config.rfile = os.path.join(config.dir+"/ranks","rank_list_%s.txt" % name)
            config.atraining = os.path.join(config.dir+"/held_out",name)
            rl,inp,order,freq,enorig = get_rdata(config,flex,elex,ttype=config.eval_set)
            en,gid = inp

            ## main decode function
            _decode_single(rl,en,aligner,gid,config.dir,self.logger)
            ## change results file
            swap_results(config.dir,name)


## factory

def Decoder(config):
    """Factory for finding decoder class

    :param config: the main configuration 
    """
    dclass = DECODERS.get(config.decoder,None)
    if dclass == None:
        raise ValueError('Uknown decoder type: %s' % dclass)
    return dclass

    
## C METHODS

cdef int _decode_single(np.ndarray rl,np.ndarray en, ## the rank and english data
                             WordModel aligner, ## the main model
                             np.ndarray[dtype=np.int32_t,ndim=1] gid,
                             str directory,
                             object logger
    ) except -1:
    """Rank/decode for a single dataset

    :param rl: the rank list 
    """
    cdef int size = en.shape[0]
    cdef int rlen = rl.shape[0]
    cdef int i,j
    cdef RankStorage storage = RankStorage.load_empty(size,rlen)
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos
    stime = time.time()
    
    ## ranker loop 
    for i in range(size):
        aligner._rank(en[i],rl,ranks[i])
        for j in range(rlen):
            if ranks[i][j] == gid[i]:
                gold_pos[i] = j
                
    logger.info('Ranked items in %f seconds, now scoring' % (time.time()-stime))
    try: storage.compute_score(directory,'baseline')
    except Exception, e: logger.error(e,exc_info=True)


## CLI INFORMATION

def argparser():
    """return an aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v

    usage = """python -m zubr rankdecoder [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser 
    
def params():
    """main parameters for running the aligners and/or aligner experiments

     :rtype: tuple
    :returns: description of option types with name, list of options 
    """
    #from zubr.SymmetricAligner import params
    from zubr.Alignment import params as aparams
    aligner_group,aligner_param = aparams()
    aligner_group["RankDecoder"] = "Settings for the rank decoder"

    ## import the neural settings
    from zubr.NeuralModels import params as nparams
    n_group,n_param = nparams()

    options = [
        ("--rfile","rfile",'',"str",
         "file with all semantic items to rank [default='']","RankDecoder"),
        ("--ranksize","ranksize",10,int,
         "rank size to evaluate [default=10]","RankDecoder"),
        ("--pseudolex","pseudolex",False,"bool",
         "add f words mapped to themselves to training  [default=False]","RankDecoder"),
        ("--termbaseline","termbaseline",False,"bool",
         "run the term baseline for comparison [default=False]","RankDecoder"),
        ("--tlambda","tlambda",0.0,"float",
         "lambda value for term baseline [default=0.0]","RankDecoder"),
        ("--baselineonly","baselineonly",False,"bool",
         "stop after running baseline [default=False]","RankDecoder"),
        ("--eval_set","eval_set",'valid',"str",
         "the evaluation set [default='validation']","RankDecoder"),
        ("--decoder","decoder",'single',"str",
         "The type of decoder to use [default='single']","RankDecoder"),
        ("--from_model","from_model",'',"str",
         "Run from an existing model [default='']","RankDecoder"), 
    ]

    options += aligner_param
    options += n_param
    return (aligner_group,options)


def main(argv):
    """The main execution link 

    :param argv: the cli input or a config 
    :rtype: list or zubr.util.config.ConfigAttrs
    :returns: None 
    """
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv)
        logging.basicConfig(level=logging.DEBUG)

    ## reload existing model

    ## try execution 
    try:
        dclass = Decoder(config)
        decoder = dclass.load_model(config)

        if config.termbaseline:
            decoder.logger.info('Running the term baseline...')

            ## make sure that test option is selected
                                    
            btime = time.time()
            run_baseline(config)

            ## log results 
            decoder.logger.info('Finished running baseline in %f seconds' % (time.time()-btime))
            #decoder.logger.info('Now exiting...' % (time.time()-btime))
            exit()
            
        ## train the underlying aligner model
        #decoder.train()
        if not config.from_model: 
            decoder.train(config=config)
            ## try to backup after training 
            try:
                decoder.model.backup(config.dir)
            except:
                pass 

        if config.rfile:
            decoder.rank(config)
    except Exception,e:
        traceback.print_exc(file=sys.stdout)
    finally:
        pass 
            
if __name__ == "__main__":
    main(sys.argv[1:])
