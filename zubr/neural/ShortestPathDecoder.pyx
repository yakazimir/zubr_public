# cython: profile=True
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""
import shutil
import os
import re
import copy
import logging
import traceback
import pickle
import time
import sys
import gzip
import datetime
from collections import defaultdict
from zubr.util import ConfigObj,ConfigAttrs
from libc.stdlib cimport malloc, free
from cython cimport wraparound,boundscheck,cdivision
from zubr.neural.util import oov_graph
from zubr.Dataset cimport RankStorage
from zubr.ExecutableModel cimport ExecutableModel
from zubr.ExecutableModel import ExecuteModel
from zubr.util.config import ConfigAttrs
from zubr.Graph cimport WordGraph,DirectedAdj,Path
import numpy as np
cimport numpy as np
from zubr.neural.Seq2Seq import NeuralModel
from zubr.util.decoder_util import *
from zubr.util.aligner_util import *
from zubr.neural.util import pad_input

from zubr.GraphDecoder cimport (
    GraphDecoderBase,
    KBestTranslations,
    SequencePath,
    SequenceBuffer
)    

from zubr.neural.Seq2Seq cimport (
    Seq2SeqModel,
    Seq2SeqLearner,
    SymbolTable,
    EncoderDecoder,
    RNNResult,
    EncoderInfo,
)

from zubr.neural._dynet cimport (
    ComputationGraph,
    get_cg,
    RNNState,
    Expression,
    log,
    concatenate_cols,
    select_cols,
    softmax,
    select_rows,
    inputVector,
    concatenate
)

## this somehow conflicts with dynet, uses numpy variants 
#from libc.math cimport log
#from libc.math cimport isinf as cisinf

cdef class NeuralSPDecoderBase(GraphDecoderBase):
    """Neural shortest path decoder base"""
    
    def train(self,config):
        """Train the underlying neural model

        :param config: the main configuration 
        :rtype config: zubr.util.config.ConfigAttrs
        :rtype: None 
        """
        self.logger.info('Training the underlying model...')
        self.learner.train(config)

    @classmethod
    def load_model(cls,config):
        """Load a neural decoder from file 

        :param config: the global configuration 
        """
        raise NotImplementedError()

    ## backup protocol

    def backup(self,wdir):
        """Backup the model to file 

        :param wdir: the working directory, place to back up 
        """
        raise NotImplementedError

    @classmethod
    def load_backup(cls,config):
        """Load a backed up decoder from file 

        :param config: the global configuration object 
        :type config: zubr.util.config.ConfigAttrs
        :rtype: NeuralSPDecoder
        """
        raise NotImplementedError
        
cdef class NeuralSPDecoder(NeuralSPDecoderBase):
    """A shortest path decoder"""

    def __init__(self,learner,graph,edge_labels,edge_map,config):
        """Create a GreedPathDecoder instance

        :param learner: the underlying neural model and trainer
        """
        self.learner = learner
        self.graph = graph
        self.edge_labels = edge_labels
        self.edge_map = edge_map
        self._config = config 

    @classmethod
    def from_config(cls,config):
        """Build a neural decoder from configuration 

        :param config: the global configuration 
        """
        cdef Seq2SeqLearner learner
        
        ## setup the learner 
        learner = Seq2SeqLearner.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels = load_graph(config,learner.stable.dec_map)
        #oov_graph(smap)
        graph = WordGraph.from_adj(edges,spans,size)
        
        return cls(learner,graph,smap,edge_labels,config)

    ## main decoder entry point
    
    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset using the neural shortest path 

        :param config: the main configuration
        """
        ##model
        cdef Seq2SeqLearner learner = self.learner
        cdef EncoderDecoder model = <EncoderDecoder>learner.model
        cdef ComputationGraph cg = learner.cg
        ## lexicons
        cdef dict edict = learner.stable.enc_map
        cdef dict fdict = learner.stable.dec_map
        
        ## graph stuff
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef dict emap = self.edge_map

        ## data related
        cdef np.ndarray en,enorig
        cdef int k = config.k
        cdef int graph_beam = config.graph_beam

        ## OOV settings
        cdef bint ignore_oov = config.ignore_oov
        cdef bint bias_match = config.bias_match

        en,enorig,rmap,gid = get_decoder_data(config,fdict,edict,ttype=config.eval_set)
        pad_input(en,model._eend)
        
        self.logger.info('Decoding with k=%d, eval_set=%s, dsize=%d, ignore oov=%s, num jobs=%d' %\
                             (k,config.eval_set,en.shape[0],ignore_oov,config.num_jobs))
                        
        ## run the decoder
        try: 
            score_dataset(model,
                              cg,
                              en,
                              enorig,
                              0,
                              k,
                              graph_beam,
                              graph,
                              edge_labels,
                              emap,
                              rmap,
                              gid,
                              self.logger,
                              ignore_oov,
                              bias_match,
                              directory=config.dir,
                              trace=config.trace
                              )
            
        except Exception,e:
            self.logger.info(e,exc_info=True)

    ## back up protocols

    def backup(self,wdir):
        """Back up the given decoder to file 

        :param wdir: the working directory 
        :rtype: None 
        """
        stime = time.time()
        decoder_dir = os.path.join(wdir,"neural_decoder")

        ## check if the
        if os.path.isdir(decoder_dir):
            self.logger.info('Already backed up, skipping...')
            self.learner.backup(wdir)
            return

        os.mkdir(decoder_dir)

        ## backup the learner
        self.learner.backup(wdir)

        ## back up the graph stuff
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_dir,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_dir,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump(self.edge_map,info)

        ### backup the config
        self._config.print_to_yaml(decoder_dir)
        
        ## log the time
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load a decoder from file 

        :param config: the configuration object 
        :rtype config: zubr.
        """
        decoder_dir = os.path.join(config.dir,"neural_decoder")
        decoder_model = os.path.join(config.dir,"neural_model")
        stime = time.time()
        odir = config.dir 

        ## load the word graph 
        graph = WordGraph.load_backup(config)

        ## load the configuration
        dconfig = ConfigAttrs()
        dconfig.restore_old(decoder_dir)
        #dconfig.dir = config.dir
        dconfig.dir = decoder_dir

        ## get the edge map
        dinfo = os.path.join(decoder_dir,"decoder_info.p")
        with gzip.open(dinfo,'rb') as info:
            edge_map = pickle.load(info)

        ## components
        labels = os.path.join(decoder_dir,"edge_labels.npz")
        archive = np.load(labels)
        elabels = archive["arr_0"]


        ## load the leaner
        dconfig.dir = odir 
        learner = Seq2SeqLearner.load_backup(dconfig)

        return cls(learner,graph,elabels,edge_map,dconfig)

    @classmethod
    def load_model(cls,config):
        """Load a neural decoder from file 

        :param config: the global configuration 
        """
        cdef Seq2SeqLearner learner

        ## the learner 
        learner = Seq2SeqLearner.load_backup(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels = load_graph(config,learner.stable.dec_map)
        graph = WordGraph.from_adj(edges,spans,size)
        
        return cls(learner,graph,smap,edge_labels)

cdef class ExecutableNeuralDecoder(NeuralSPDecoderBase):
    """Neural decoder for executable models"""
    
    def __init__(self,learner,graph,edge_labels,edge_map,executor,config):
        """Creates an ExecutableNeuralDecoder instance 

        :param model: the underlying translation model 
        :param graph: the underlying word graph 
        :param symbols: the underlying symbol/path table 
        :param edge_map: point from edges to positions in symbol table 
        :param oov: the out of vocabular map
        :param executor: the execution model 
        """
        self.learner = learner
        self.graph       = graph
        self.edge_labels = edge_labels
        self.edge_map    = edge_map
        self.executor    = <ExecutableModel>executor
        self._config     = config

    def __exit__(self,exc_type, exc_val, exc_tb):
        ## make sure to shut off the executor 
        self.executor.exit()

    def exit(self):
        self.executor.exit()

    @classmethod 
    def load_model(cls,config):
        """Load a neural decoder from file 

        :param config: the global configuration 
        """
        cdef Seq2SeqLearner learner

        ## the learner 
        learner = Seq2SeqLearner.load_backup(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels = load_graph(config,learner.stable.dec_map)
        graph = WordGraph.from_adj(edges,spans,size)

        ## executable model
        eclass = ExecuteModel(config)
        executor = eclass.from_config(config)
        
        return cls(learner,graph,smap,edge_labels,executor,config)

    @classmethod
    def from_config(cls,config):
        """Build a neural decoder from configuration 

        :param config: the global configuration 
        """
        cdef Seq2SeqLearner learner
        
        ## setup the learner 
        learner = Seq2SeqLearner.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels = load_graph(config,learner.stable.dec_map)
        #oov_graph(smap)
        graph = WordGraph.from_adj(edges,spans,size)

        ## executeable model
        eclass = ExecuteModel(config)
        executor = eclass.from_config(config)

        return cls(learner,graph,smap,edge_labels,executor,config)
    
    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset using the neural shortest path 

        :param config: the main configuration
        """
        ##model
        cdef Seq2SeqLearner learner = self.learner
        cdef EncoderDecoder model = <EncoderDecoder>learner.model
        cdef ComputationGraph cg = learner.cg
        ## lexicons
        cdef dict edict = learner.stable.enc_map
        cdef dict fdict = learner.stable.dec_map
        
        ## graph stuff
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef dict emap = self.edge_map

        ## data related
        cdef np.ndarray en,enorig
        cdef int k = config.k
        cdef int graph_beam = config.graph_beam

        ## executable model
        cdef ExecutableModel executor =  <ExecutableModel>self.executor

        ## oov settings
        cdef bint ignore_oov = config.ignore_oov
        cdef bint bias_match = config.bias_match

        #en,enorig,rmap,gid = get_decoder_data(config,fdict,edict,ttype=config.eval_set)
        en,enorig,rmap,gid,grep = get_executor_data(config,fdict,edict,ttype=config.eval_set)
        pad_input(en,model._eend)
        
        self.logger.info('Decoding with k=%d, eval_set=%s, dsize=%d, ignore oov=%s' %\
                             (k,config.eval_set,en.shape[0],ignore_oov))
                        
        ## run the decoder
        try:
            execute_dataset(model,executor,
                                cg,
                                en,
                                enorig,
                                grep,
                                0,
                                k,
                                graph_beam,
                                graph,
                                edge_labels,
                                emap,
                                rmap,
                                gid,
                                self.logger,
                                ignore_oov,
                                bias_match,
                                directory=config.dir,
                                trace=config.trace
                                )

        except Exception,e:
            self.logger.info(e,exc_info=True)


    ## back up protocols

    def backup(self,wdir):
        """Back up the given decoder to file 

        :param wdir: the working directory 
        :rtype: None 
        """
        stime = time.time()
        decoder_dir = os.path.join(wdir,"neural_decoder")

        ## check if the
        if os.path.isdir(decoder_dir):
            self.logger.info('Already backed up, skipping...')
            self.learner.backup(wdir)
            return
        os.mkdir(decoder_dir)

        ## backup the learner
        self.learner.backup(wdir)

        ## back up the graph stuff
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_dir,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_dir,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump(self.edge_map,info)

        ### backup the config
        self._config.print_to_yaml(decoder_dir)

        ## log the time
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load a decoder from file 

        :param config: the configuration object 
        :rtype config: zubr.
        """
        decoder_dir = os.path.join(config.dir,"neural_decoder")
        decoder_model = os.path.join(config.dir,"neural_model")
        stime = time.time()
        odir = config.dir 

        ## load the word graph 
        graph = WordGraph.load_backup(config)

        ## load the configuration
        dconfig = ConfigAttrs()
        dconfig.restore_old(decoder_dir)
        #dconfig.dir = config.dir
        dconfig.dir = decoder_dir

        ## get the edge map
        dinfo = os.path.join(decoder_dir,"decoder_info.p")
        with gzip.open(dinfo,'rb') as info:
            edge_map = pickle.load(info)

        ## components
        labels = os.path.join(decoder_dir,"edge_labels.npz")
        archive = np.load(labels)
        elabels = archive["arr_0"]

        ## load the leaner
        dconfig.dir = odir 
        learner = Seq2SeqLearner.load_backup(dconfig)

        ## executeable model
        eclass = ExecuteModel(config)
        executor = eclass.from_config(config)
                        
        return cls(learner,graph,elabels,edge_map,executor,dconfig)

## polyglot executable graph decoder

cdef class PolyglotExecutableNeuralDecoder(ExecutableNeuralDecoder):
    """Polyglot decoder for executable models"""

    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset using the neural shortest path 

        :param config: the main configuration
        """
        ##model
        cdef Seq2SeqLearner learner = self.learner
        cdef EncoderDecoder model = <EncoderDecoder>learner.model
        cdef ComputationGraph cg = learner.cg
        ## lexicons
        cdef dict edict = learner.stable.enc_map
        cdef dict fdict = learner.stable.dec_map
        
        ## graph stuff
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef dict emap = self.edge_map

        ## data related
        cdef np.ndarray en,enorig
        cdef int k = config.k
        cdef int graph_beam = config.graph_beam

        ## executable model
        cdef ExecutableModel executor =  <ExecutableModel>self.executor

        ## oov stuff
        cdef bint ignore_oov = config.ignore_oov
        cdef bint bias_match = config.bias_match

        en,enorig,rmap,gid,grep,langs = get_executor_data(config,fdict,edict,ttype=config.eval_set,poly=True)
        pad_input(en,model._eend)
        
        self.logger.info('Decoding with k=%d, eval_set=%s, dsize=%d, skip oov=%s' %\
                             (k,config.eval_set,en.shape[0],ignore_oov))

        ## run the decoder
        try: 
            pexecute_dataset(model,executor,
                                cg,
                                en,enorig,grep,
                                langs,
                                0,k,graph_beam,
                                graph,edge_labels,
                                emap,rmap,gid,
                                self.logger,
                                ignore_oov,
                                bias_match,
                                directory=config.dir,
                                trace=config.trace
                                )
        except Exception,e:
            self.logger.info(e,exc_info=True)


cdef class PolyglotSPNeuralDecoder(NeuralSPDecoderBase):
    """Decoding with polyglot datasets"""

    def __init__(self,learner,graph,edge_labels,edge_map,lang_starts,config):
        """Creates an ExecutableNeuralDecoder instance 

        :param model: the underlying translation model 
        :param graph: the underlying word graph 
        :param symbols: the underlying symbol/path table 
        :param edge_map: point from edges to positions in symbol table 
        :param oov: the out of vocabular map
        :param lang_starts: the starting points in the graph for the different output languages
        """
        self.learner = learner
        self.graph       = graph
        self.edge_labels = edge_labels
        self.edge_map    = edge_map
        self.langs = lang_starts
        self._config = config

    @classmethod
    def from_config(cls,config):
        """Build a neural decoder from configuration 

        :param config: the global configuration 
        """
        cdef Seq2SeqLearner learner
        
        ## setup the learner 
        learner = Seq2SeqLearner.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels,langs = load_graph(config,learner.stable.dec_map,poly=True)
        #oov_graph(smap)
        graph = WordGraph.from_adj(edges,spans,size)

        return cls(learner,graph,smap,edge_labels,langs,config)

    @classmethod
    def load_model(cls,config):
        """Load a neural decoder from file 

        :param config: the global configuration 
        """
        cdef Seq2SeqLearner learner

        ## the learner 
        learner = Seq2SeqLearner.load_backup(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels,langs = load_graph(config,learner.stable.dec_map,poly=True)
        graph = WordGraph.from_adj(edges,spans,size,langs)
        
        return cls(learner,graph,smap,edge_labels,langs)

    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset using the neural shortest path 

        :param config: the main configuration
        """
        ##model
        cdef Seq2SeqLearner learner = self.learner
        cdef EncoderDecoder model = <EncoderDecoder>learner.model
        cdef ComputationGraph cg = learner.cg

        ## symbol table
        cdef SymbolTable stable = learner.stable
        
        ## lexicons
        cdef dict edict = learner.stable.enc_map
        cdef dict fdict = learner.stable.dec_map
        
        ## graph stuff
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef dict emap = self.edge_map

        ## data related
        cdef np.ndarray en,enorig
        cdef int k = config.k
        cdef int graph_beam = config.graph_beam
        cdef bint spec_lang = config.spec_lang
        cdef dict lang_map = self.langs

        ## oov settings
        cdef bint ignore_oov = config.ignore_oov
        cdef bint bias_match = config.bias_match

        #lang_codes = {i:fdict[i] for i in lang_map}
        lang_codes = {i:fdict.get(i,0) for i in lang_map}

        en,enorig,rmap,gid,langs = get_decoder_data(config,
                                                        fdict,
                                                        edict,
                                                        ttype=config.eval_set,
                                                        poly=True)
        pad_input(en,model._eend)
        
        self.logger.info('Polyglot decoding with k=%d, eval_set=%s, dsize=%d, #langs=%d,skip oov=%s' %\
                             (k,config.eval_set,en.shape[0],len(lang_map),ignore_oov))
                        
        ## run the decoder
        try: 
            score_poly(model,cg,
                           en,enorig,
                           langs,spec_lang,
                           lang_map,
                           k,graph_beam,
                           graph,edge_labels,
                           emap,rmap,gid,
                           self.logger,
                           ignore_oov,
                           bias_match,
                           directory=config.dir,
                           lang_codes=lang_codes,
                           trace=config.trace
                           )
        except Exception,e:
            self.logger.info(e,exc_info=True)


    ## backup protocol

    def backup(self,wdir):
        """Back up the given decoder to file 

        :param wdir: the working directory 
        :rtype: None 
        """
        stime = time.time()
        decoder_dir = os.path.join(wdir,"neural_decoder")

        ## check if the
        if os.path.isdir(decoder_dir):
            self.logger.info('Already backed up, skipping...')
            self.learner.backup(wdir)

            return
        os.mkdir(decoder_dir)

        ## backup the learner
        self.learner.backup(wdir)

        ## back up the graph stuff
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_dir,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_dir,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump({"edges":self.edge_map,"langs":self.langs},info)

        ### backup the config
        self._config.print_to_yaml(decoder_dir)
        
        ## log the time
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))            

    @classmethod
    def load_backup(cls,config):
        """Load a decoder from file 

        :param config: the configuration object 
        :rtype config: zubr.
        """
        decoder_dir = os.path.join(config.dir,"neural_decoder")
        decoder_model = os.path.join(config.dir,"neural_model")
        stime = time.time()
        odir = config.dir 

        ## load the word graph 
        graph = WordGraph.load_backup(config)

        ## load the configuration
        dconfig = ConfigAttrs()
        dconfig.restore_old(decoder_dir)
        dconfig.dir = decoder_dir

        ## get the edge map
        dinfo = os.path.join(decoder_dir,"decoder_info.p")
        with gzip.open(dinfo,'rb') as info:
            #edge_map = pickle.load(info)
            items = pickle.load(info)
            edge_map = items["edges"]
            langs = items["langs"]

        ## components
        labels = os.path.join(decoder_dir,"edge_labels.npz")
        archive = np.load(labels)
        elabels = archive["arr_0"]

        ## load the leaner
        dconfig.dir = odir 
        learner = Seq2SeqLearner.load_backup(dconfig)
        
        return cls(learner,graph,elabels,edge_map,langs,dconfig)

## concurrent decoders

cdef class ConcurrentNeuralSPDecoder(NeuralSPDecoder):
    """Decoder that concurrently decodes data by splitting data and copying model. 
    For a full description, see zubr.GraphDecoder.ConcurrentWordModelDecoder

    """
    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset asynchronously by splitting the data into n pieces

        :param config: the main configuration, which specifies the data location 
        and the number of jobs to run.
        """
        raise NotImplementedError

    cpdef _setup_jobs(self,config):
        """Setup the infrastructure needed to run concurrent jobs

        :param config: the main configuration
        """
        cdef RankStorage storage

        # ## dump the current model 
        dtime = time.time()
        self.logger.info('Backing up the current model...')
        self.backup(config.dir)

        # ## split the dataset up into n jobs and run the jobs 
        self.logger.info('Now setting up the jobs infrastructure...')
        it = time.time()
        _,rsize = setup_jobs(config)
        self.logger.info('Copied and setup up in %s seconds' % str(time.time()-it))

        # # ## score the joined together item
        merge = os.path.join(config.dir,"merged_ranks.txt")
        with open(merge) as mc: flen = sum([1 for i in mc])
        self.logger.info('file length: %d, rank_size=%d' % (flen,rsize))


        try: 
            #storage = RankStorage.load_from_file(merge,flen,rsize)
            storage = RankStorage.load_from_file(merge,flen,config.k,rsize)
            storage.logger.info('Loaded new storage instance...')
            self.score_ranks(config.dir,config.k,storage,config=config)

            ## backup the storage (since it takes so long to read from file)
            storage.backup(config.dir,name=config.eval_set)

        except Exception,e:
            self.logger.error(e,exc_info=True)

    def score_ranks(self,wdir,k,ranks,config=None):
        """Score the ranks according to the type of decoder 

        :param ranks: the formatted rank storage item 
        """
        raise NotImplementedError

cdef class NeuralPolyglotConcurrentDecoder(ConcurrentNeuralSPDecoder):
    """Concurrent model for polyglot neural models """

    def __init__(self,learner,graph,edge_labels,edge_map,lang_starts,config):
        """Creates an ExecutableNeuralDecoder instance 

        :param model: the underlying translation model 
        :param graph: the underlying word graph 
        :param symbols: the underlying symbol/path table 
        :param edge_map: point from edges to positions in symbol table 
        :param oov: the out of vocabular map
        :param lang_starts: the starting points in the graph for the different output languages
        """
        self.learner = learner
        self.graph       = graph
        self.edge_labels = edge_labels
        self.edge_map    = edge_map
        self.langs = lang_starts
        self._config = config

    @classmethod
    def from_config(cls,config):
        """Build a neural decoder from configuration 

        :param config: the global configuration 
        """
        cdef Seq2SeqLearner learner
        
        ## setup the learner 
        learner = Seq2SeqLearner.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels,langs = load_graph(config,learner.stable.dec_map,poly=True)
        graph = WordGraph.from_adj(edges,spans,size)

        return cls(learner,graph,smap,edge_labels,langs,config)

    def score_ranks(self,wdir,k,ranks,config=None):
        """Score the ranks according to the type of decoder 

        :param ranks: the formatted rank storage item 
        """
        cdef Seq2SeqLearner learner = self.learner
        cdef dict edict = learner.stable.enc_map
        cdef dict fdict = learner.stable.dec_map

        ## this should be fixed
        self.logger.info('Scoring the new rank list')
        stuff  = get_decoder_data(config,fdict,edict,ttype=config.eval_set,poly=True)
        ranks.compute_poly_score(wdir,stuff[-1],stuff[2],k,dtype='baseline')

    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset asynchronously by splitting the data into n pieces

        :param config: the main configuration, which specifies the data location 
        and the number of jobs to run.
        """
        self._setup_jobs(config)
        
    load_backup = PolyglotSPNeuralDecoder.load_backup

    def backup(self,wdir):
        """Back up the given decoder to file 

        :param wdir: the working directory 
        :rtype: None 
        """
        stime = time.time()
        decoder_dir = os.path.join(wdir,"neural_decoder")

        ## check if the
        if os.path.isdir(decoder_dir):
            self.logger.info('Already backed up, skipping...')

            ## backup the learner
            self.learner.backup(wdir)
            
            return

        ## making a new directory 
        os.mkdir(decoder_dir)

        ## backup the learner
        self.learner.backup(wdir)

        ## back up the graph stuff
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_dir,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_dir,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump({"edges":self.edge_map,"langs":self.langs},info)

        ### backup the config
        self._config.print_to_yaml(decoder_dir)
        
        ## log the time
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))


cdef class NeuralConcurrentDecoder(ConcurrentNeuralSPDecoder):
    """Monolingual concurrent neumodel """

    load_backup = NeuralSPDecoder.load_backup

    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset asynchronously by splitting the data into n pieces

        :param config: the main configuration, which specifies the data location 
        and the number of jobs to run.
        """
        self._setup_jobs(config)

    def score_ranks(self,wdir,k,ranks,config=None):
        """Score the ranks according to the type of decoder 

        :param ranks: the formatted rank storage item 
        """
        ranks.compute_mono_score(wdir,k,'baseline')



## helper classes

cdef class NeuralSequencePath(SequencePath):
    """Specialized sequence path for neural models"""
    
    def __init__(self,np.ndarray path,
                     np.ndarray eseq,
                     np.ndarray useq,
                     np.ndarray node_scores,
                     np.ndarray state_seq,
                     double score,int size
        ):
        """Create a NeuralSequencePath instance 

        :param path: the graph path 
        :param eseq: the encoded translation model sequence 
        :param useq: the unicode translation 
        :param node_scores: the scores of each node 
        :param state_seq: the RNN states at each point 
        :param score: the sequence score 
        :param size: the size of the translation 
        """
        self.seq  = path
        self.eseq = eseq
        self.score = score
        self.state_seq = state_seq 
        self.size = size
        self.node_scores = node_scores
        self._translation = useq

    cdef np.ndarray eos_encoding(self,int eos):
        """Convert the encoded representation to one padded with <EOS> symbols

        :param eos: the identifier of the eos symbol
        
        """
        cdef np.ndarray[ndim=1,dtype=np.int32_t] eseq = self.eseq
        return np.insert(np.insert(eseq,0,eos),sequence.shape[0]+1,eos)
    
## c methods

@boundscheck(False)
cdef int execute_dataset(EncoderDecoder model,
                            ExecutableModel executor,
                            ComputationGraph cg,
                            np.ndarray dataset,
                            np.ndarray dataset_orig,
                            np.ndarray grep,
                            int start_node,
                            int k,int graph_beam,
                            WordGraph graph,
                            int[:] edge_labels,
                            dict emap,
                            dict rmap,
                            int[:] gid,
                            object logger,
                            bint ignore_oov,
                            bint bias_match,
                            directory=None,
                            trace=False
        ) except -1:
    """Applies an executable decoder model to some data"""
    ## graph properties 
    cdef DirectedAdj adj = graph.adj
    cdef int gsize = graph.num_nodes

    ## output paths 
    cdef KBestTranslations paths
    #cdef SequencePath path
    cdef NeuralSequencePath path
    cdef bint gold_found

    ## unicode output
    cdef unicode esentence,utranslation,gtranslation
    
    ### 
    cdef int identifier,num,i,size = dataset.shape[0]
    cdef double st = time.time()

    ## storage items
    cdef int rsize = len(rmap)
    ## rank object
    cdef RankStorage storage = RankStorage.load_empty(size,k+1)
    #cdef RankStorage storage = RankStorage.load_empty(size,k)
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos
    cdef dict other_gold = storage.other_gold

    ## about gold stuff 
    cdef int gnoi = 0
    cdef set repeats

    cdef dict word_lookup

    ## start the counter 
    stime = time.time()

    ## go through the data
    for i in range(size):
        other_gold[i] = set()

        ## check if fold answer is found
        gold_found = False

        ## the input 
        esentence = np.unicode(dataset_orig[i])
        word_lookup = {o:k+1 for k,o in enumerate(esentence.split())}

        gtranslation = np.unicode(grep[gid[i]])

        try: 
            paths = shortest_path(adj,edge_labels,
                                esentence,
                                word_lookup,
                                gsize,
                                k+1,
                                graph_beam,
                                start_node,
                                dataset[i],
                                model,
                                cg,
                                emap,
                                ignore_oov,
                                bias_match)
            
        except Exception,e:
            logger.error(e,exc_info=True)

        for num,path in enumerate(paths):
            utranslation = path.translation_string
            if utranslation not in rmap:
                rmap[utranslation] = len(rmap)

            identifier = rmap[utranslation]
            ranks[i][num] = identifier

            ## checks if the denotation matches the generated representation 
            matching_den = executor.evaluate(gtranslation.replace('@@ ',''),
                                                 utranslation.replace('@@ ',''))
            
            if not gold_found and matching_den:
                gold_found  = True
                gold_pos[i] = num

            elif matching_den:
                other_gold[i].add(num)

        if not gold_found:
            gold_pos[i] = k
            ranks[i][k] = gid[i]
            gnoi += 1

    logger.info('Decoded and scored %d sentences in %s seconds (%d not in beam)' %\
                    (size,str(time.time()-stime),gnoi))

    if directory:
        #storage.compute_score(directory,'baseline')
        ## printipy out the new rank link
        storage.compute_mono_score(directory,k,'baseline')

@boundscheck(False)
cdef int pexecute_dataset(EncoderDecoder model,
                            ExecutableModel executor,
                            ComputationGraph cg,
                            np.ndarray dataset,
                            np.ndarray dataset_orig,
                            np.ndarray grep,
                            np.ndarray langs,
                            int start_node,
                            int k,int graph_beam,
                            WordGraph graph,
                            int[:] edge_labels,
                            dict emap,
                            dict rmap,
                            int[:] gid,
                            object logger,
                            bint ignore_oov,
                            bint bias_match,
                            directory=None,
                            trace=False,
       ) except -1:
    """Applies an executable decoder model to some data"""
    ## graph properties 
    cdef DirectedAdj adj = graph.adj
    cdef int gsize = graph.num_nodes

    ## output paths 
    cdef KBestTranslations paths
    #cdef SequencePath path
    cdef NeuralSequencePath path
    cdef bint gold_found

    ## unicode output
    cdef unicode esentence,utranslation,gtranslation
    
    ### 
    cdef int identifier,num,i,size = dataset.shape[0]
    cdef double st = time.time()

    ## storage items
    cdef int rsize = len(rmap)
    ## rank object
    cdef RankStorage storage = RankStorage.load_empty(size,k+1)
    #cdef RankStorage storage = RankStorage.load_empty(size,k)
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos
    cdef dict other_gold = storage.other_gold

    ## about gold stuff 
    cdef int gnoi = 0
    cdef set repeats

    ##
    cdef dict word_lookup

    ## start the counter 
    stime = time.time()

    ## go through the data
    for i in range(size):
        other_gold[i] = set()

        ## check if fold answer is found
        gold_found = False

        ## the input sentence 
        esentence = np.unicode(dataset_orig[i])
        word_lookup = {o:k+1 for k,o in enumerate(esentence.split())}

        gtranslation = np.unicode(grep[gid[i]])

        try: 
            paths = shortest_path(adj,edge_labels,
                                esentence,
                                word_lookup,
                                gsize,
                                k+1,
                                graph_beam,
                                start_node,
                                dataset[i],
                                model,
                                cg,
                                emap,
                                ignore_oov,
                                bias_match)
            
        except Exception,e:
            logger.error(e,exc_info=True)

        for num,path in enumerate(paths):
            utranslation = path.translation_string
            if utranslation not in rmap:
                rmap[utranslation] = len(rmap)

            identifier = rmap[utranslation]
            ranks[i][num] = identifier

            ## checks if the denotation matches the generated representation 
            #matching_den = executor.evaluate(gtranslation,utranslation)
            matching_den = executor.evaluate(gtranslation.replace('@@ ',''),
                                                 utranslation.replace('@@ ',''))
            
            if not gold_found and matching_den:
                gold_found  = True
                gold_pos[i] = num

            elif matching_den:
                other_gold[i].add(num)

        if not gold_found:
            gold_pos[i] = k
            ranks[i][k] = gid[i]
            gnoi += 1

    logger.info('Decoded and scored %d sentences in %s seconds (%d not in beam)' %\
                    (size,str(time.time()-stime),gnoi))

    if directory:
        storage.compute_poly_score(directory,langs,rmap,k,dtype='baseline',exc=True)
        #storage.backup(directory,name=etype)

        
                            
## polyglot solve

@boundscheck(False)
cdef int score_poly(EncoderDecoder model,
                           ComputationGraph cg,
                           np.ndarray dataset,
                           np.ndarray dataset_orig,
                           np.ndarray langs,
                           bint spec_lang,
                           dict lang_map,
                           int k,int graph_beam,
                           WordGraph graph,
                           int[:] edge_labels,
                           dict emap,
                           dict rmap,
                           int[:] gid,
                           object logger,
                           bint ignore_oov,
                           bint bias_match,
                           directory=None,
                           lang_codes={},
                           trace=False
        ) except -1:
    """Runs the decoder on a dataset
    
    Note: There is a lot of redundancy with what's in Zubr.GraphDecoder, 
    should be merged at some point.

    :param model: the underlying neural model 
    :param cg: the dynet computation graph 
    :param dataset: the dataset to decoder 
    :param start_node: the place to start when decoding 
    :param graph: the underlying decoder graph 
    :param edge_labels: label edges in the graph 
    :param emap: the edge symbol map 
    :param rmap: the rank list in map form 
    :param gid: the gold ids for the dataset 
    :param logger: a logger instance 
    """
    cdef int i,size = dataset.shape[0]
    cdef unicode esentence,utranslation

    ## graph items
    cdef DirectedAdj adj = graph.adj
    cdef int gsize = graph.num_nodes
    
    ## output paths 
    cdef KBestTranslations paths
    #cdef SequencePath path
    cdef NeuralSequencePath path
    cdef bint gold_found

    ### 
    cdef int identifier,num
    cdef double st = time.time()

    ## language stuff
    cdef int source

    ## storage items
    cdef int rsize = len(rmap)
    #cdef RankStorage storage = RankStorage.load_empty(size,rsize)
    cdef RankStorage storage = RankStorage.load_empty(size,k+1)
    cdef int[:,:] ranks = storage.ranks

    ## about gold stuff 
    cdef int[:] gold_pos = storage.gold_pos
    cdef int gnoi = 0
    cdef set repeats

    ##
    cdef dict word_lookup
    cdef object lang_code
    cdef bint tracing = False

    ## tracing
    if trace and directory:
        tracing = True 
        trace_file = open(os.path.join(directory,"traces.txt"),'w')

    ### log information about whether the language is being set 
    logger.info('Decoding polyglot dataset, set_lang=%s ...' % str(spec_lang))

    ## start the counter 
    stime = time.time()

    ## go through the data
    for i in range(size):
        gold_found = False

        ## trace the number 
        if tracing:
            logger.info('Processing sentence: %d' % i)

        ## the input sentence
        esentence = np.unicode(dataset_orig[i])
        word_lookup = {o:k+1 for k,o in enumerate(esentence.split())}

        ## where to start the search?
        source = 0 if not spec_lang else lang_map[langs[i]][1]
        lang_code = None if source == 0 else lang_codes[langs[i]]

        ## compute the k-best paths
        try: 
            paths = shortest_path(adj,edge_labels,
                                esentence,
                                word_lookup,
                                gsize,
                                k+1,
                                graph_beam,
                                source,
                                dataset[i],
                                model,
                                cg,
                                emap,
                                ignore_oov,
                                bias_match,
                                pre=lang_code,
                                )
        except Exception,e:
            logger.error(e,exc_info=True)

        repeats = set()

        for num,path in enumerate(paths):
            
            ## the unicode translation string and identifier in rank
            utranslation = path.translation_string
            utranslation = utranslation if source == 0 else u"%s %s" % (langs[i],utranslation)
            identifier = rmap.get(utranslation,-1)

            if identifier in repeats:
                logger.info('Repeat detected: %s' % utranslation)
            repeats.add(identifier)

            ranks[i][num] = identifier
            if identifier == gid[i]:
                gold_found  = True
                gold_pos[i] = num

                ## trace output 
                if tracing: print >>trace_file,num     
                                                
                
            ## unknown output? 
            if identifier == -1:
                logger.warning('Unknown output: %s for input %d' % (utranslation,i))

        if not gold_found:
            gold_pos[i] = rsize 
            #gold_pos[i] = k
            #ranks[i][k] = gid[i]
            gnoi += 1

            ## trace the unknown stuff 
            if tracing: print >>trace_file,-1

    # # # log time information 
    logger.info('decoded and scored %d sentences in %s seconds (%d not in beam)' %\
                    (size,str(time.time()-st),gnoi))

    # # ## score if desired
    if directory:
        storage.compute_poly_score(directory,langs,rmap,k,dtype='baseline')
        if tracing: trace_file.close()


@boundscheck(False)
cdef int score_dataset(EncoderDecoder model,
                           ComputationGraph cg,
                           np.ndarray dataset,
                           np.ndarray dataset_orig,
                           int start_node,
                           int k,int graph_beam,
                           WordGraph graph,
                           int[:] edge_labels,
                           dict emap,
                           dict rmap,
                           int[:] gid,
                           object logger,
                           bint ignore_oov,
                           bint bias_match,
                           directory=None,
                           trace=False,
        ) except -1:
    """Runs the decoder on a dataset
    
    Note: There is a lot of redundancy with what's in Zubr.GraphDecoder, 
    should be merged at some point.

    :param model: the underlying neural model 
    :param cg: the dynet computation graph 
    :param dataset: the dataset to decoder 
    :param start_node: the place to start when decoding 
    :param graph: the underlying decoder graph 
    :param edge_labels: label edges in the graph 
    :param emap: the edge symbol map 
    :param rmap: the rank list in map form 
    :param gid: the gold ids for the dataset 
    :param logger: a logger instance 
    """
    cdef int i,size = dataset.shape[0]
    cdef unicode esentence,utranslation

    ## graph items
    cdef DirectedAdj adj = graph.adj
    cdef int gsize = graph.num_nodes
    
    ## output paths 
    cdef KBestTranslations paths
    #cdef SequencePath path
    cdef NeuralSequencePath path
    cdef bint gold_found

    ### 
    cdef int identifier,num
    cdef double st = time.time()

    ## word lookup
    cdef dict word_lookup

    ## storage items
    cdef int rsize = len(rmap)
    cdef RankStorage storage = RankStorage.load_empty(size,rsize)
    cdef int[:,:] ranks = storage.ranks

    ## about gold stuff 
    cdef int[:] gold_pos = storage.gold_pos
    cdef int gnoi = 0
    cdef set repeats
    cdef bint tracing = False

    ## tracing
    if trace and directory:
        tracing = True 
        trace_file = open(os.path.join(directory,"traces.txt"),'w')

    ## start the counter 
    stime = time.time()

    ## go through the data
    for i in range(size):

        ## selector to see if gold was found 
        gold_found = False

        if tracing:
            logger.info('Processing sentence: %d' % i)

        ## the input sentence in unicode form
        esentence = np.unicode(dataset_orig[i])
        word_lookup = {o:k+1 for k,o in enumerate(esentence.split())}

        ## compute the k-best paths
        try: 
            paths = shortest_path(adj,edge_labels,
                                esentence,
                                word_lookup,
                                gsize,
                                k+1,
                                graph_beam,
                                start_node,
                                dataset[i],
                                model,
                                cg,
                                emap,
                                ignore_oov,
                                bias_match)
            
        except Exception,e:
            logger.error(e,exc_info=True)

        repeats = set()

        for num,path in enumerate(paths):
            
            ## the unicode translation string and identifier in rank
            utranslation = np.unicode(path.translation_string)
            identifier =  rmap.get(utranslation,-1)
            ranks[i][num] = identifier

            ## check if any repeats come up 
            if identifier in repeats:
                logger.info('Repeat detected: %s' % utranslation)
                continue 
            repeats.add(identifier)

            ## is a gold item?
            if identifier == gid[i]:
                gold_found = True
                gold_pos[i] = num

                if tracing: print>>trace_file,num

            ## unknown output? 
            if identifier == -1:
                logger.warning('Unknown output: %s for input %d' % (utranslation,i))

        if not gold_found:
            gold_pos[i] = (rsize - 1)
            ranks[i][rsize-1] = gid[i]
            gnoi += 1
            if tracing: print>>trace_file,-1

    ## log the information
    logger.info('Decoded and scored %d sentences in %s seconds (%d not in beam)' %\
                    (size,str(time.time()-stime),gnoi))

    if directory:
        storage.compute_mono_score(directory,k,'baseline')
        if tracing: trace_file.close()

@boundscheck(False)
#@wraparound(False)
@cdivision(True)
cdef KBestTranslations shortest_path(DirectedAdj adj,int[:] labels, # graph components
                                      unicode uinput,
                                      dict eloc,
                                      int size, ## the number of edges 
                                      int k, ## the size of the
                                      int graph_beam,
                                      int starting_point,
                                      np.ndarray[ndim=1,dtype=np.int32_t] english,
                                      EncoderDecoder model,
                                      ComputationGraph cg,
                                      dict emap,
                                      bint ignore_oov,
                                      bint bias_match,
                                      object pre=None,
                                      set blocks=set(), ## global blocks in the graph                                      
                                      ):
    """Extracts the k single source shortest paths

    How does this work? 

    -- First of all, the 

    :param adj: the graph adjacency 
    :param labels: the edge labels for the graph 
    :param uninput: the english input in unicode form 
    :param size: the size of the graph 
    :param k: the number of translations to generate 
    :param starting_point: the place to start in the graph 
    :param english: the english encoded representation  
    :param model: the neural network seq2seq model 
    :param cg: the dynet computation graph 
    :param emap: the lexical/edge map 
    :param ignore_oov: skip over the OOV, don't penalize the score, assign -log(0)
    :param bias_match: assign -log(0) to oov matching words 
    """
    ## graph components 
    cdef int[:] edges   = adj.edges
    cdef int[:,:] spans = adj.node_spans

    ## k best list 
    cdef np.ndarray sortest_k_best,A = np.ndarray((k,),dtype=np.object)
    cdef np.ndarray prev_states,root_states,spur_state
    #cdef SequencePath top_candidate,previous,spur_path,new_path,recent_path
    cdef NeuralSequencePath top_candidate,previous,spur_path,new_path,recent_path
    cdef SequenceBuffer B = SequenceBuffer()
    cdef SequenceBuffer AS = SequenceBuffer()

    ## the different sequences involved -
    cdef np.ndarray[ndim=1,dtype=np.int32_t]  prev_seq, spur_seq,total_path,root_path,other_path,prev_eseq
    cdef np.ndarray[ndim=1,dtype=np.int32_t]  total_eseq,spur_eseq
    #cdef np.ndarray[ndim=1,dtype=np.double_t] sput_score,root_score
    cdef np.ndarray[ndim=1,dtype=np.double_t] prev_score,spur_score,total_node_score
    cdef np.ndarray prev_trans,spur_trans,total_trans,total_state
    cdef float root_score
    
    cdef dict spurs_seen = <dict>defaultdict(set)
    ## equality lookup
    cdef int[:] equal_path = np.zeros((k,),dtype=np.int32)

    ## input information
    cdef isize = english.shape[0]

    cdef int current_k,i,prev_size
    cdef int root_len
    cdef int root_outgoing,block_len
    cdef double total_score,top_score,prev_num

    ## block information
    cdef set root_block,observed_spurs,ignored_nodes,ignored_edges
    cdef empty_scores = np.zeros((isize,),dtype='d')

    ## encoder and decoder
    cdef RNNState s
    cdef Expression init_embed = model.get_init_embedding(cg)
    cdef Expression input_mat

    ## english input
    cdef list encoding
    cdef int start_word
    cdef Expression model_expr
    cdef double model_score
    cdef EncoderInfo e
    cdef int last_word

    ## renew the computation graph
    cg.renew(False,False,None)

    ## encoder stuff 
    e = model.encode_input(english,cg)
    encoding = e.encoded
    input_mat = concatenate_cols(encoding)

    ## initial decoder state 
    s = model.get_dec_init(cg)
    last_word = model._dend

    ## generate a new state? 
    if pre:
        s = model.append_state(s,e,input_mat,model._dend,cg)
        last_word = pre

    ## add option of adding another input to initialize (e.g., lang type)
    
    ## initial best path
    previous = faster_best_path(starting_point,
                                    0.0,
                                    0,
                                    graph_beam,
                                    edges,
                                    spans,
                                    labels,
                                    size,
                                    #encoding,
                                    eloc,
                                    e,
                                    model,
                                    cg,
                                    s,
                                    #model._dend,
                                    last_word,
                                    input_mat,
                                    emap,
                                    ignore_oov,
                                    bias_match)

    A[0] = previous
    current_k = 1
    #AS.push(previous.score,previous)

    # ## need to fill in 
    while True:
        if current_k >= k: break
        equal_path[:] = 0

        ## initialize what to ignore
        ignore_nodes = set()
        ignore_edges = set()

        ## add global blocks 
        ignore_edges = ignore_edges.union(blocks)

        ## previous size and actual sequence
        prev_size  = previous.size
        prev_seq   = previous.seq
        prev_score = previous.node_scores
        prev_eseq  = previous.eseq
        prev_trans = previous.translation
        prev_num   = previous.score
        prev_states = previous.state_seq

        ## rerun through neural model to get real loss
        
        ## add it to a special item
        # model_expr = model.get_loss(previous.eos_encoding(model._eend))
        # model_score = model_expr.value()
        AS.push(previous.score,previous)
        #AS.push(model_score,previous)

        ## go through each part of the last best item 
        for i in range(1,prev_size):
            root_path = prev_seq[:i]
            root_states = prev_states[:i]
            root_len = root_path.shape[0]

            ## root score
            root_score = prev_score[i-1]

            ## number of outgoing
            root_outgoing = (spans[root_path[-1]][-1]+1)-spans[root_path[-1]][0]
            root_block = set()

            ## iterate through other k-best candidates 
            for j in range(0,current_k):

                ## recent path from k-best list so far 
                recent_path = A[j]
                other_path = recent_path.seq

                ## easy case, previous best path is the most recent 
                if j == (current_k - 1):
                    ignore_edges.add((other_path[i-1],other_path[i]))
                    root_block.add(other_path[i])

                ## starting point 
                elif i == 1 and root_path[0] == other_path[0]:
                    ignore_edges.add((other_path[i-1],other_path[i]))
                    root_block.add(other_path[i])
                    equal_path[j] = 1

                ## previously equal, new part equal?
                elif equal_path[j] == 1 and root_path[-1] == other_path[i-1]:
                    ignore_edges.add((other_path[i-1],other_path[i]))
                    root_block.add(other_path[i])

                ## not equal anymore 
                elif equal_path[j] == 1 and i > 1:
                    equal_path[j] = 0

            ##
            ignore_nodes.add(root_path[-1])
            block_len = len(root_block)

            ## check if all edges have been exhausted    
            if block_len == root_outgoing: continue
            ## check if restriction have already been looked at (Lawler's rule)
            observed_spurs = spurs_seen[tuple(root_path)]
            if block_len <= len(observed_spurs): continue
            observed_spurs.update(root_block)

            ## starting word
            start_word = model._dend if i == 1 else prev_eseq[i-2]

            ## do the next best search
            spur_path = faster_best_path(root_path[-1],
                                    root_score,
                                    i,
                                    graph_beam,
                                    edges,spans,labels,size,
                                    eloc,
                                    e,
                                    model,
                                    cg,
                                    root_states[-1], ## 
                                    start_word,
                                    input_mat,
                                    emap,
                                    ignore_oov,
                                    bias_match,
                                    ignored_edges=ignore_edges)

            ## sput path and score 
            spur_seq   = spur_path.seq
            spur_score = spur_path.node_scores
            spur_state = spur_path.state_seq

            ## eseq and translation 
            spur_eseq = spur_path.eseq
            spur_trans = spur_path.translation

            ## glue together
            total_path = np.concatenate((root_path,spur_seq[1:]),axis=0)
            total_score = spur_path.score
            total_node_score = np.concatenate((prev_score[:i-1],spur_score))

            ## concacenate encoded sequence            
            total_eseq = np.concatenate((prev_eseq[:i-1],spur_eseq))
            total_trans = np.concatenate((prev_trans[:i-1],spur_trans))
            total_state = np.concatenate((prev_states,spur_state[1:]),axis=0)

            ## new candidate path
            new_path = NeuralSequencePath(total_path,
                                              total_eseq,
                                              total_trans,
                                              total_node_score,
                                              total_state,
                                              total_score,
                                              total_path.shape[0])
            B.push(total_score,new_path)

        if B.empty(): break

        ## get the top candidate
        top_candidate = B.pop()
        top_score = top_candidate.score

        A[current_k] = top_candidate
        current_k += 1
        previous = top_candidate
        
    sortest_k_best = AS.kbest(current_k)
    return KBestTranslations(english,uinput,A,current_k)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef NeuralSequencePath faster_best_path(int source,
                                    double init_score,int start_len,
                                    int graph_beam,
                                    int[:] edges,int[:,:] spans,
                                    int[:] labels,int size,
                                    dict ewords,                                    
                                    EncoderInfo einput,
                                    ## model
                                    EncoderDecoder model,
                                    ComputationGraph cg,
                                    RNNState s,
                                    int embed,
                                    Expression input_mat,
                                    dict emap,
                                    bint ignore_oov,
                                    bint bias_match,
                                    set ignored_edges=set(),## edges to ignore
                                    set ignored_nodes=set(),## nodes to ignore
         ):
    """ 
    Compute a greedy path from a given point in the underlying component graph 

    :param source: the source node in the graph 
    :param edges: the graph edges 
    :param spans: the span container 
    :param einput: the encoded english input 
    :param model: the encoder decoder neural model 
    :param dec_start_state: the decoder initial state 
    :param emap: map of the graph edges 
    """
    cdef dict state_map,encoding_map
    cdef int i,j,w,start,end,node,cword
    cdef double cdef_float_infp = float('+inf')

    ## the resulting scores and sequences
    cdef np.ndarray[ndim=1,dtype=np.int32_t] end_seq
    cdef np.ndarray[ndim=1,dtype=np.double_t] fnode_scores
    cdef np.ndarray[ndim=1,dtype=np.int32_t] encoded_sequence
    cdef np.ndarray unicode_seq,state_seq

    ## score information during search 
    cdef double *d            = <double *>malloc(size*sizeof(double))
    cdef int *p               = <int *>malloc(size*sizeof(int))
    cdef double *node_scores  = <double *>malloc(size*sizeof(double))
    cdef int *out_seq         = <int *>malloc(size*sizeof(int))

    ## edge labels 
    cdef int *best_labels     = <int *>malloc(size*sizeof(int))
    cdef int *final_labels    = <int *>malloc(size*sizeof(int))

    ## length size
    cdef double *nseq_len     = <double *>malloc(size*sizeof(double))

    ## sequence scores at each node 
    cdef int *final_seq      = <int *>malloc(size*sizeof(int))
    
    ## neural stuff
    cdef RNNResult node_result
    cdef Expression nout,node_prob,reduced_out
    cdef RNNState node_state,new_state
    cdef double score,nscore

    ## labels
    cdef list candidate_labels,adj_list 
    cdef np.ndarray[ndim=1,dtype=np.double_t] probs,new_vals
    cdef double[:] best_scores
    cdef int pos,csize,current,seq_len
    cdef double final_score,o
    cdef double ss = float(start_len)
    cdef long[:] sorted_args
    cdef int current_item,current_index,arg_position,alen

    ### copying information
    cdef bint has_copy
    cdef dict lookup = einput.lookup
    cdef Expression attention_weights,new_softmax,selected
    cdef int word,lsize,copy_val
    cdef double tval,amax
    cdef unicode oov_word
    
    ## output

    ## put initial state
    state_map = {i:None for i in range(source,size)}
    encoding_map = {i:None for i in range(source,size)}
    state_map[source] = s
    encoding_map[source] = embed

    try:

        ## initialize the count objects
        with nogil:
            for i in range(source,size):
                d[i]            = cdef_float_infp
                p[i]            = -1
                out_seq[i]      = -1
                node_scores[i]  = cdef_float_infp
                best_labels[i]  = 0
                final_seq[i]    = -1
                nseq_len[i]     = 0.0

            ## initialize the source score 
            d[source] = init_score
            nseq_len[source] = ss

        ## now interate through the graph from source -> end

        for i in range(source,size-1):

            ## start and end in the graph adjacency 
            start = spans[i][0]
            end = spans[i][1]

            ## early pruning, dead ends  
            if i > source and p[i] == -1: continue

            ## compute a score using the neural network 
            node_result = model.get_dec_distr_scratch(state_map[i],einput,input_mat,encoding_map[i],cg)
            nout = node_result.probs
            node_state = node_result.state

            ## attention and copying stuff
            #has_copy = node_result.copy
            has_copy = model.copy
            
            ## the output labels at this point in the adjacency 
            candidate_labels = [j if j != -1 else 0 for j in labels[start:end+1]]
            adj_list = range(start,end+1)

            ## number of items
            pos = graph_beam - 1
            if pos >= (end+1-start): pos = len(candidate_labels)-1

            ## softmax over ONLY the output labels in the graph adjacency
            selected = select_rows(nout,candidate_labels)
            
            ## model with copies 
            if has_copy:

                lsize = len(candidate_labels)
                attention_weights = node_result.attention_scores
                new_softmax = softmax(<Expression>concatenate([selected,attention_weights]))
                new_vals = new_softmax.npvalue()

                ## update the individual items
                for word in range(lsize):

                    ## the identifier of the copy item 
                    copy_val = candidate_labels[word]
                    
                    ## check if copied value has a higher score 
                    if copy_val in lookup:

                        ## compare the normal write score with (=tval) with the copy score (=amax)
                        tval = new_vals[word]
                        amax = new_vals[<int>lsize+lookup[copy_val]]

                        ## if copying is better than assign score 
                        if tval <= amax: new_vals[word] = amax

                    ## check if OOV and matches somethign in source
                    if copy_val == 0:

                        ## find the word associated with this OOV edge
                        j = adj_list[word]
                        oov_word = emap[(i,edges[j])]

                        if not ignore_oov and oov_word in ewords:

                            ## similarly check if 
                            tval = new_vals[word]
                            amax = new_vals[<int>lsize+ewords[oov_word]]

                            ## if copying oov to value in source is better, do it
                            if tval <= amax and not bias_match: new_vals[word] = amax
                            elif bias_match: new_vals[word] = 0.0

                        ## skip over the word 
                        elif ignore_oov:
                            new_vals[word] = 0.0

                    ## skip over the OOV words

                ## create the probability vector to sort below 
                probs = -np.log(new_vals[:lsize])

            ## model without copies 
            else:
                node_prob = softmax(selected)
                probs = <Expression>(-log(node_prob)).npvalue()

            ## sort output by ids 
            sorted_args = np.argsort(probs)
            alen = sorted_args.shape[0]
            current_item  = 0
            current_index = 0
            arg_position  = -1

            while True:
                #arg_position = sorted_args[current_item]
                if current_index >= alen: break 
                arg_position = sorted_args[current_index]
                score = d[i]+probs[arg_position]

                j = adj_list[arg_position]
                cword = candidate_labels[arg_position]
                node = edges[j]

                ## check for blocks 
                if i == source and (i,node) in ignored_edges:
                    current_index += 1
                    continue

                ## normalize score by length 
                ## add 2 because you have <EOS> on both sides 
                if node == (size - 1): score = score/(nseq_len[i]+2.0)

                ## add new scrore 
                if d[node] > score:
                    d[node] = score
                    p[node] = i
                    encoding_map[node] = cword
                    state_map[node] = node_state
                    best_labels[node] = cword
                    nseq_len[node] = nseq_len[i]+1.0

                current_index += 1
                current_item += 1
                if current_item >= pos: break

        ## put together the final path
        out_seq[0] = size-1
        node_scores[0] = d[size-1]
        final_score = d[size-1]
        current = size-1
        seq_len = 1
        final_seq[0] = best_labels[size-1]

        while True:
            current = p[current]
            out_seq[seq_len] = current
            node_scores[seq_len] = d[current]
            final_seq[seq_len]   = best_labels[current]
            if current <= source: break
            seq_len += 1

        ## the final sequence
        end_seq = np.array([i for i in out_seq[:seq_len+1]],dtype=np.int32)[::-1]

        ## unicode sequence
        unicode_seq =  np.array([emap[(end_seq[i],end_seq[i+1])] for i in range(seq_len-1)],dtype=np.object)

        ## only go to :seq_len because last symbol is a special graph token *END*
        encoded_sequence = np.array([i for i in final_seq[:seq_len]],dtype=np.int32)[::-1][:seq_len-1]

        ## sequence scores
        fnode_scores = np.array([o for o in node_scores[:seq_len+1]],dtype='d')[::-1]

        ## state sequence
        state_seq = np.array([state_map[i] for i in end_seq],dtype=np.object)

        return NeuralSequencePath(
            end_seq,
            encoded_sequence,
            unicode_seq,
            fnode_scores,
            state_seq,
            final_score,
            seq_len+1            
        )

    finally:
        free(d)
        free(p)
        free(node_scores)
        free(out_seq)
        free(best_labels)
        free(final_labels)
        free(final_seq)
        free(nseq_len)

## factories

DECODERS = {
    "sp"  : NeuralSPDecoder, ## monolingual neural decoder
    "psp" : PolyglotSPNeuralDecoder, ## polyglot non-executable decoder 
    "ex"  : ExecutableNeuralDecoder, ## executable decoder 
    "pex" : PolyglotExecutableNeuralDecoder, ## polyglot executable neural decoder
    ## concurrent models
    "con_sp"   : NeuralConcurrentDecoder,
    "con_poly" : NeuralPolyglotConcurrentDecoder,
}

def Decoder(dtype):
    """Factory method for retrieving a decoder class 

    :param dtype: the type of decoder to use 
    :raises: ValueError
    """
    dclass = DECODERS.get(dtype,None)
    if dclass is None:
        raise ValueError('Uknown type of decoder: %s' % dclass)
    return dclass

def params():
    """Parameters and settings for building neural graph decoder

    :type: tuple 
    """
    from zubr.GraphDecoder import params as dparams
    
    groups = {}
    groups["NeuralDecoder"] = "Settings for neural decoder"
    groups["ConcurrentDecoder"] = "Settings for concurrent decoder"
    from zubr.ExecutableModel import params as eparams
    e_group,e_params = eparams()

    
    options = [
        ("--k","k",100,"int",
         "The size of shortest paths to generate  [default=100]","NeuralDecoder"),
        ("--decoder","decoder","sp","str",
         "The type of neural decoder to use [default='sp']","NeuralDecoder"),
        ("--model_loc","model_loc","","str",
         "The location of an existing model [default='']","NeuralDecoder"),
        ("--graph_beam","graph_beam",2,int,
         "The size of the beam for graph search [default=2]","NeuralDecoder"),
        ("--eval_set","eval_set",'valid',"str",
         "The dataset to evaluate on [default='valid']","NeuralDecoder"),
        ("--num_jobs","num_jobs",5,"int",
         "The number of concurrent jobs to run [default=2]","ConcurrentDecoder"),
        ("--jlog","jlog",'',"str",
         "The logger path for running subjobs [default='']","ConcurrentDecoder"),         
        ("--spec_lang","spec_lang",False,"bool",
         "Decoding according to specified language (for polyglot models) [default=False]","NeuralDecoder"),
        ("--ignore_oov","ignore_oov",False,"bool",
         "Skip over oov items and assign log(1.0) score [default=False]","NeuralDecoder"),
        ("--bias_match","bias_match",False,"bool",
         "Bias matching words by assigning zero score [default=False]","NeuralDecoder"),
        ("--run","run",'',"str",
         "Run an existing model [default='']","NeuralDecoder"),
        ("--trace","trace",False,"bool",
         "Trace the decoder steps [default=False]","NeuralDecoder"),
        ("--from_neural","from_neural","","str",
         "Pointer to the neural model [default='']","NeuralDecoder"),
        ("--more_train","more_train",False,"bool",
         "Train an existing model for more time [default=False]","NeuralDecoder"),                                                     
    ]

    dgroup,doptions = dparams()
    options += doptions
    options += e_params
    groups.update(dgroup)
    groups.update(e_group)
    return (groups,options)

def argparser():
    """Create an argument parser for this module"""
    from zubr import _heading
    from _version import __version__ as v

    usage = """python -m zubr neuralpath [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser


def main(argv):
    """Main entry point for using the neural shortest path decoder 

    :param argv: a configuration object or string cli input 
    :rtype: None 
    """

    ## setup the configuration 
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])

    ## logging information
    if config.jlog:
        logging.basicConfig(filename=config.jlog,level=logging.DEBUG)
    else: logging.basicConfig(level=logging.DEBUG)

    #print "begin=%d" % config.dec_state_size
        
    try:

        dclass = Decoder(config.decoder)

        if config.dir:
            ## make a rerun script
            rerun_script(config)
            npath = os.path.join(config.dir,"neural_model")

        else:
            npath = 'empty'

        odir = config.dir

        ## use a model to run 
        if config.from_neural and os.path.isdir(config.from_neural):

            ## set the working directory as place where model is
            config.dir = os.path.dirname(config.from_neural)
            odir = config.dir

            ## load model and dump it into a new directory 
            decoder = dclass.from_config(config)

            ### swap the model
            decoder.logger.info('Trying to swap the loaded model..')
            decoder.learner.swap_model(config)

            ## making a new directory
            ty = 'test' if not config.more_train else 'retrain'
            dirname = datetime.datetime.fromtimestamp(time.time()).strftime(ty+'%Y-%m-%d-%H:%M:%S')
            config.dir = os.path.join(odir,dirname)
            os.mkdir(config.dir)

            ## train the model for more time 
            # if config.more_train:
            #     config.wdir = os.path.dirname(config.atraining)
            #     decoder.train(config)

            ## back up this new model 
            decoder.backup(config.dir)

        ## train an existing model more
        # elif config.more_train and os.path.isdir(config.more_train):
        #     pass
        
        ## run an existing model 
        elif config.model_loc and os.path.isdir(config.model_loc):
            config.dir = config.model_loc
            decoder = dclass.load_backup(config)

        ## train a model from scratch 
        else:
            decoder = dclass.from_config(config)
            decoder.train(config)
            
        ## decode some data
        decoder.decode_data(config)

        ## change back directory
        if odir: config.dir = odir

    except Exception,e:
       traceback.print_exc(file=sys.stdout)

    ## exit the decoder (important for the executable models)  
    finally:
        
        ## backup the model (if not backed up already)
        try: 
            decoder.backup(config.dir)
        except Exception,e:
            decoder.logger.error('Error backing up the final model at %s!' % config.dir)
            decoder.logger.error(e,exc_info=True)

        ## exit the model 
        try:
            decoder.exit()
        except Exception,e:
            pass

## alias for main 
run_decoder = main

if __name__ == "__main__":
    main(sys.argv[1:])
