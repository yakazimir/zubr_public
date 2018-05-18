# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""
import logging
import sys
import os
import time
import gzip
import shutil
import cPickle as pickle
import traceback
import subprocess 
from collections import defaultdict
import numpy as np
cimport numpy as np
from zubr.util import ConfigObj,ConfigAttrs
from zubr.ZubrClass cimport ZubrSerializable
from heapq import heappush, heappop, nsmallest
from zubr.util.decoder_util import *
from zubr.util.aligner_util import *
from zubr.SymmetricAlignment cimport SymmetricWordModel,SymAlign,Phrases
from zubr.Alignment cimport WordModel,Alignment
from zubr.Graph cimport WordGraph,DirectedAdj,Path
from zubr.ExecutableModel cimport ExecutableModel
from zubr.ExecutableModel import ExecuteModel
from zubr.Dataset cimport RankStorage
from libc.stdlib cimport malloc, free
from libc.math cimport log,isinf
from cython cimport wraparound,boundscheck,cdivision
from zubr.NeuralModels cimport FeedForwardLM


cdef class GraphDecoderBase(ZubrSerializable):
    """Base for grapher decoder model"""

    @classmethod
    def from_config(cls,config):
        """Loads a graph decoder from a configuration 

        :param configuration: the target configuration
        :returns: graph decoder instance
        """
        raise NotImplementedError

    def backup(self,wdir):
        """Back up a graph decoder model

        :param wdir: the working directory where to place model 
        :rtype: None 
        """
        raise NotImplementedError

    @classmethod
    def load_backup(cls,config):
        """Load a graph decoder model

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        :returns: a graph decoder instance 
        """
        raise NotImplementedError

    ## default exit
    
    def exit(self):
        """Method for exiting the decoder, in case decoder is running subprocessed 
        
        :note: passes by default 
        """
        pass

    def train(self,config):
        """Train the underlying word model

        :rtype: None 
        """
        raise NotImplementedError

    ## c methods

    cpdef int decode_data(self,object config) except -1:
        """Decode a given dataset input

        :param config: the configuration, which has pointer to data 
        """
        pass
    

cdef class WordModelDecoder(GraphDecoderBase):

    cdef KShortestPaths decode_input(self,int[:] decode_input,int k):
        """Decode a given encoded input and find k-shortest paths

        :param decode_input: the input vector encoding a sequence 
        :param k: the number of shortest paths to return
        """
        raise NotImplementedError

    # cpdef int decode_data(self,object config) except -1:
    #     """Decode a given dataset input

    #     :param config: the configuration, which has pointer to data 
    #     """
    #     pass

    cpdef KBestTranslations translate(self,dinput,int k):
        """Translate a given input

        :param dinput: the decoder input 
        :type dinput: basestring 
        """
        raise NotImplementedError

    def train(self,config):
        """Train the underlying word model

        :rtype: None 
        """
        self.logger.info('Training the underlying model...')
        self.model.train(config)

    cdef np.ndarray _encode_input(self,text_input):
        """Encode a given text input into the representation of the underlying model

        :param text_input: the input text 
        """
        raise NotImplementedError

    ## alignment methods

    cdef SymAlign align(self,int[:] output_v,int[:] input_v,str heuristic):
        """Performs a symmetric alignment 

        :param output_v: the output vector encoding
        :param intput_v: the input vector encoding 
        :param heuristic: the alignment heuristic
        """
        cdef SymmetricWordModel model = <SymmetricWordModel>self.model
        return model._align(output_v,input_v,np.insert(input_v,0,0),heuristic=heuristic)    


    ## implement enter and exit protocols for using with `with` construct

    def __enter__(self):
        return self 

    def __exit__(self,exc_type, exc_val, exc_tb):
        pass

    ## properties

    property elex:
        """The English side vocabulary and lexicon"""
        
        def __get__(self):
            """Returns the english side lexicon

            :rtype: dict
            """
            cdef SymmetricWordModel model = <SymmetricWordModel>self.model
            return model.elex

    property flex:
        """The English side vocabulary and lexicon"""
        
        def __get__(self):
            """Returns the english side lexicon

            :rtype: dict
            """
            cdef SymmetricWordModel model = <SymmetricWordModel>self.model
            return model.flex

    property elen:
        """The size of the english side vocabular"""
        
        def __get__(self):
            """Returns the size of the english side lexicon

            :rtype: int 
            """
            cdef SymmetricWordModel model = <SymmetricWordModel>self.model
            return model.elen

    property flen:
        """The size of the foreign side vocabulary and lexicon"""
        
        def __get__(self):
            """Returns the foreign side vocabular size

            :rtype: int
            """
            cdef SymmetricWordModel model = <SymmetricWordModel>self.model
            return model.flen

cdef class WordGraphDecoder(WordModelDecoder):

    """Class for graph decoders that involve directed acyclic graphs (DAGs), also called word graphs

    -- the main idea: 

    The search space (set of output representations) for the decoder is represented as a DAG. 
    For an input, the graph edges are annotated with weights using information from a translation 
    model, and  candidates representations are generated by finding the best (or shortest) paths in graph. 

    The goal is to generate k-best paths that can be used for some other downstream process or learning 
    (e.g., discriminative traing..) 

    """

    def __init__(self,model,graph,edge_labels,edge_map):
        """Create a word graph decoder instance 


        :param model: the underlying translation model 
        :param graph: the underlying word graph 
        :param symbols: the underlying symbol/path table 
        :param edge_map: point from edges to positions in symbol table 
        :param oov: the out of vocabular map
        """
        self.model       = <SymmetricWordModel>model
        self.graph       = graph
        self.edge_labels = edge_labels
        self.edge_map    = edge_map

    @classmethod
    def from_config(cls,config):
        """Loads a dag decoder from configuration 

        :param configuration: the target configuration 
        :type: zubr.util.config.ConfigAttrs
        :returns: a dag decoder instance 
        """
        ## setup the underlying model
        model = SymmetricWordModel.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels = load_graph(config,model.ftoe.flex)
        graph = WordGraph.from_adj(edges,spans,size)
        
        return cls(model,graph,smap,edge_labels)

    cpdef int decode_data(self,object config) except -1:
        """Decode a given dataset input

        :param config: the configuration, which has pointer to data 
        :returns: Nothing 
        """
        ## translation model 
        cdef SymmetricWordModel model = <SymmetricWordModel>self.model
        cdef dict edict = model.ftoe.elex
        cdef dict fdict = model.ftoe.flex

        ## graph components and information
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef int[:] gid
        cdef dict emap = self.edge_map

        ## data infor 
        cdef np.ndarray en,enorig
        cdef int k = config.k

        ## build the data
        en,enorig,rmap,gid = get_decoder_data(config,fdict,edict,ttype=config.eval_set)
        self.logger.info('Decoding with k=%d, eval_set=%s, dsize=%d' % (k,config.eval_set,en.shape[0]))

        ## run the decoder 
        score_dataset(model,
                          en,enorig,
                          0,k,
                          self.graph,edge_labels,
                          emap,rmap,gid,
                          self.logger,
                          config.eval_set,
                          directory=config.dir)
        
                          #,hoffman=config.hoffman)

    cpdef KBestTranslations translate(self,dinput,int k):
        """Translate a given input

        -- This currently has some very weird error for some datasets, 
        e.g., Java, complains about a error with malloc and a linked
        list. Will post more later (this error came up in relation to 
        a test somewhere in zubr/zubr/test/

        :param dinput: the decoder input 
        :type dinput: basestring 
        :param k: the size of translated outputs to return
        :rtype k: int
        """
        cdef unicode uinput = to_unicode(dinput)
        cdef np.ndarray[ndim=1,dtype=np.int32_t] encoded = self._encode_input(uinput)
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef DirectedAdj adj = graph.adj
        cdef int gsize = graph.num_nodes
        cdef SymmetricWordModel model = <SymmetricWordModel>self.model
        cdef dict emap = self.edge_map
        cdef WordModel aligner = <WordModel>model.ftoe
        cdef double[:,:] table = aligner.make_table()
        
        return shortest_path(adj,edge_labels,
                            uinput,
                            gsize,
                            k+1,
                            0,
                            encoded,
                            #model,
                            aligner,
                            table,
                            emap)
    
    cdef np.ndarray _encode_input(self,text_input):
        """Encode a text input into the representation of the underlying model

        :param text_input: the input text 
        """
        cdef SymmetricWordModel model = <SymmetricWordModel>self.model
        cdef dict elex = model.elex
        cdef unicode w,uinput

        uinput = to_unicode(text_input)
        return np.array([elex.get(w,-1) for w in uinput.split()],dtype=np.int32)

    ## backup protocol

    def backup(self,wdir):
        """Back up a graph decoder model

        :param wdir: the working directory where to place model 
        :rtype: None 
        """
        stime = time.time()
        decoder_path = os.path.join(wdir,"graph_decoder")
        if os.path.isdir(decoder_path):
            self.logger.info('Already backed up, skipping...')
            return

        os.mkdir(decoder_path) 
        ## back up the underlying translation model 
        self.model.backup(wdir)
        ## back up the graph model 
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_path,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_path,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump(self.edge_map,info)

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))
        
    @classmethod
    def load_backup(cls,config):
        """Load a graph decoder model

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        :returns: a graph decoder instance 
        """
        decoder_path = os.path.join(config.dir,"graph_decoder")
        stime = time.time()

        ## the other components
        model = SymmetricWordModel.load_backup(config)
        graph = WordGraph.load_backup(config)

        ## get the edge map 
        dinfo = os.path.join(decoder_path,"decoder_info.p")
        with gzip.open(dinfo,'rb') as info:
            edge_map = pickle.load(info)

        ## components
        labels = os.path.join(decoder_path,"edge_labels.npz")
        archive = np.load(labels)
        elabels = archive["arr_0"]

        instance = cls(model,graph,elabels,edge_map)
        instance.logger.info('Loaded backup in %s seconds' % str(time.time()-stime))
        return instance
    
    def __reduce__(self):
        ## pickle implementation 
        return WordGraphDecoder,(self.model,self.graph,self.edge_labels,self.edge_map)


cdef class PolyglotWordDecoder(WordModelDecoder):
    """A decoder for multiple languages """

    def __init__(self,model,graph,edge_labels,edge_map,lang_starts):
        """ 

        :param model: the underlying (word) translation model 
        :param edge_labels: word labels for edges in the graph 
        :param edge_map: point from edges to positions in the symbol table 
        :param lang_starts: the starting positions of difference languages 
        """
        self.model = <SymmetricWordModel>model
        self.graph = graph
        self.edge_labels = edge_labels
        self.edge_map = edge_map
        self.langs = lang_starts

    cpdef int decode_data(self,object config) except -1:
        """Decode a given dataset input

        :param config: the configuration, which has pointer to data 
        :returns: Nothing 
        """
        ## translation model 
        cdef SymmetricWordModel model = <SymmetricWordModel>self.model
        cdef dict edict = model.ftoe.elex
        cdef dict fdict = model.ftoe.flex

        ## graph components and information
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef int[:] gid
        cdef dict rmap,emap = self.edge_map
        cdef dict lang_map = self.langs
        cdef set blocks

        ## data infor 
        cdef np.ndarray en,enorig,langs
        cdef int k = config.k

        ## specify language when predicting? 
        cdef bint spec_lang = config.spec_lang

        ## build the data
        en,enorig,rmap,gid,langs = get_decoder_data(config,fdict,edict,ttype=config.eval_set,poly=True)
        self.logger.info('Decoding with k=%d, eval_set=%s, dsize=%d' % (k,config.eval_set,en.shape[0]))

        ## blocked languages
        blocks = language_blocks(config,lang_map)
        
        ## run the decoder
        try:
            score_poly(model,
                        en,
                        enorig,
                        langs,
                        k,
                        self.graph,
                        edge_labels,
                        emap,
                        rmap,
                        lang_map,
                        gid,
                        spec_lang,
                        self.logger,
                        directory=config.dir,
                        blocks=blocks,
                        )

        except Exception,e:
            self.logger.error(e,exc_info=True)
            
    @classmethod
    def from_config(cls,config):
        """Loads a dag decoder from configuration 

        :param configuration: the target configuration 
        :type: zubr.util.config.ConfigAttrs
        :returns: a dag decoder instance 
        """
        ## setup the underlying model
        model = SymmetricWordModel.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels,langs = load_graph(config,model.ftoe.flex,poly=True)
        graph = WordGraph.from_adj(edges,spans,size)
        
        return cls(model,graph,smap,edge_labels,langs)

    ## back up implementation
    ## Note: there is a lot of redundancy with the one above, this should be fixed
    def backup(self,wdir):
        """Back up a graph decoder model

        :param wdir: the working directory where to place model 
        :rtype: None 
        """
        stime = time.time()
        decoder_path = os.path.join(wdir,"graph_decoder")
        if os.path.isfile(decoder_path) or os.path.isdir(decoder_path):
            self.logger.info('Already backed up, skipping...')
            return 

        os.mkdir(decoder_path) 
        ## back up the underlying translation model 
        self.model.backup(wdir)
        ## back up the graph model 
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_path,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_path,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump({"edges":self.edge_map,"langs":self.langs},info)

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))
        
        
    @classmethod
    def load_backup(cls,config):
        """Load a graph decoder model

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        :returns: a graph decoder instance 
        """
        decoder_path = os.path.join(config.dir,"graph_decoder")
        stime = time.time()

        ## the other components
        model = SymmetricWordModel.load_backup(config)
        graph = WordGraph.load_backup(config)
        
        ## get the edge map 
        dinfo = os.path.join(decoder_path,"decoder_info.p")
        with gzip.open(dinfo,'rb') as info:
            #edge_map = pickle.load(info)
            items = pickle.load(info)
            edge_map = items["edges"]
            langs = items["langs"]

        ## components
        labels = os.path.join(decoder_path,"edge_labels.npz")
        archive = np.load(labels)
        elabels = archive["arr_0"]

        instance = cls(model,graph,elabels,edge_map,langs)
        instance.logger.info('Loaded backup in %s seconds' % str(time.time()-stime))
        return instance

    def __reduce__(self):
        ## pickle implementation
        return PolyglotWordDecoder,(self.model,self.graph,self.edge_labels,self.edge_map,self.langs)

## executable graph decoder

cdef class ExecutableGraphDecoder(WordModelDecoder):
    """A graph decoder that has an executable model (rather than a rank list)"""

    def __init__(self,model,graph,edge_labels,edge_map,executor):
        """Create an executable graph decoder 
        
        :param model: the underlying translation model 
        :param graph: the underlying word graph 
        :param symbols: the underlying symbol/path table 
        :param edge_map: point from edges to positions in symbol table 
        :param oov: the out of vocabular map
        :param executor: the execution model 
        """
        self.model       = <SymmetricWordModel>model
        self.graph       = graph
        self.edge_labels = edge_labels
        self.edge_map    = edge_map
        self.executor    = <ExecutableModel>executor

    def __exit__(self,exc_type, exc_val, exc_tb):
        ## make sure to shut off the executor 
        self.executor.exit()

    def exit(self):
        self.executor.exit()

    ## backup protocol

    def backup(self,wdir):
        """Backup the ExecutableGraphDecoder 


        :param wdir: the working directory, or place to put backup 
        :type wdir: str
        :rtype: None 
        """
        stime = time.time()
        decoder_path = os.path.join(wdir,"graph_decoder")
        if os.path.isdir(decoder_path):
            self.logger.info('Already backed up, skipping...')
            return

        os.mkdir(decoder_path) 
        ## back up the underlying translation model 
        self.model.backup(wdir)
        ## back up the graph model 
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_path,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_path,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump(self.edge_map,info)

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load an executable graph decoder backup from file

        :param config: the main configuration 
        :rtype: ExecutableGraphDecoder
        """
        decoder_path = os.path.join(config.dir,"graph_decoder")
        stime = time.time()

        ## the other components
        model = SymmetricWordModel.load_backup(config)
        graph = WordGraph.load_backup(config)
        
        ## get the edge map 
        dinfo = os.path.join(decoder_path,"decoder_info.p")
        with gzip.open(dinfo,'rb') as info:
            edge_map = pickle.load(info)

        ## components
        labels = os.path.join(decoder_path,"edge_labels.npz")
        archive = np.load(labels)
        elabels = archive["arr_0"]

        #executable model
        eclass = ExecuteModel(config)
        executor = eclass.from_config(config)

        instance = cls(model,graph,elabels,edge_map,executor)
        instance.logger.info('Loaded backup in %s seconds' % str(time.time()-stime))
        return instance
        

    cpdef int decode_data(self,object config) except -1:
        """Decode a given dataset input

        :param config: the configuration, which has pointer to data 
        :returns: Nothing 
        """
        ## translation model 
        cdef SymmetricWordModel model = <SymmetricWordModel>self.model
        cdef dict edict = model.ftoe.elex
        cdef dict fdict = model.ftoe.flex

        ## executor
        cdef ExecutableModel executor = self.executor

        ## graph components and information
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef int[:] gid
        cdef dict emap = self.edge_map

        ## data infor 
        cdef np.ndarray en,enorig,grep
        cdef int k = config.k

        ## build the data
        en,enorig,rmap,gid,grep = get_executor_data(config,fdict,edict,ttype=config.eval_set)
        self.logger.info('Decoding with k=%d, eval_set=%s, dsize=%d' % (k,config.eval_set,en.shape[0]))

        ## run the decoder 
        execute_dataset(model,
                          executor,
                          en,
                          enorig,
                          grep,
                          k,
                          self.graph,
                          edge_labels,
                          emap,rmap,gid,
                          self.logger,
                          config.eval_set,
                          config.dir)

    @classmethod
    def from_config(cls,config):
        """Load an ExecutableGraph decoder from configuration 

        :param configuration: the main configuration 
        """
        ## setup the underlying model
        model = SymmetricWordModel.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels = load_graph(config,model.ftoe.flex)
        graph = WordGraph.from_adj(edges,spans,size)
        
        ## set up the executable model 
        eclass = ExecuteModel(config)
        executor = eclass.from_config(config)

        return cls(model,graph,smap,edge_labels,executor)

    def __reduce__(self):
        ## pickle implemented
        return ExecutableGraphDecoder,(self.model,self.graph,self.edge_labels,self.edge_map)


cdef class ExecutablePolyGraphDecoder(ExecutableGraphDecoder):

    cpdef int decode_data(self,object config) except -1:
        """Decode a given dataset input 

        :param config: the main configuration for running the experiment
        """
        ## translation model 
        cdef SymmetricWordModel model = <SymmetricWordModel>self.model
        cdef dict edict = model.ftoe.elex
        cdef dict fdict = model.ftoe.flex

        ## executor
        cdef ExecutableModel executor = self.executor

        ## graph components and information
        cdef WordGraph graph = self.graph
        cdef int[:] edge_labels = self.edge_labels
        cdef int[:] gid
        cdef dict emap = self.edge_map

        ## data infor 
        cdef np.ndarray en,enorig,langs
        cdef int k = config.k

        ## build the dataset
        en,enorig,rmap,gid,grep,langs = get_executor_data(config,fdict,edict,ttype=config.eval_set,poly=True)
        self.logger.info('Decoding with k=%d, eval_set=%s, dsize=%d' % (k,config.eval_set,en.shape[0]))

        self.logger.info('about to decode the dataset')

        ## special function for poly dataset execution
        pexecute_dataset(model,
                          executor,
                          en,
                          enorig,
                          langs,
                          grep,
                          k,
                          self.graph,
                          edge_labels,
                          emap,rmap,gid,
                          self.logger,
                          config.eval_set,
                          config.dir)
    
    def __reduce__(self):
        ## pickle implementation
        return ExecutablePolyGraphDecoder(self.model,self.graph,self.edge_labels,self.edge_map)



## neural models


# cdef class NeuralShortestPath(WordModelDecoder):
#     """These graph decoder models use neural network sequence models to assign weights 
#     in the general shortest path algorithm. 
#     """
#     pass

# cdef class PolyglotNeuralShortestPath(WordModelDecoder):
#     """This class implements polyglot neural models"""
#     pass 

## Concurrent model

cdef class ConcurrentWordModelDecoder(WordModelDecoder):
    """A general decoder that decodes a number of parallel processes

    Explanation: training the word decoders is fast, since it involves
    ordinary EM training of the underlying alignment models. Even for
    hundreds of thousands of sentences, it trains in only a few minutes
    (using quite a bit of memory though, but that's another issue).  

    Decoding sentences is fairly costly, partly because of the the complexity
    of the underlying shortest path implementation and the size of the associated
    graphs in large datasets. 

    Often, these graph decoders are used to generate candidate representations that 
    are later rescored using, for example a discriminative reranker. So during training
    of such a reranker, it makes sense to store the graph decoder output in, say a text 
    file, rather than repeatedly running the decoder to generate output. 
    
    The point of this concurrent word model decoder is to train a single model, then 
    split the large dataset to be decoder into n pieces, and just decode each piece 
    asynchronously. To do this, I am using subprocess to make a number of system calls
    that call several zubr runs asynchronously. 

    This requires making n copies of the underlying translation models, and of course 
    requires n * sizeof(model) ram when running, which can be quite considerate for a
    large n, or number of concurrent process. The point being that you should be very
    careful when using this, make sure there are enough cores and memory to support n. 
    
    So the process: 1) train a single model, 2) serialize it, 3) split dataset into n
    and make n copies of the model, 4) wait until each process ends then read the rank
    output for each subprocess and join. 
    """
    cdef np.ndarray _encode_input(self,text_input):
        raise ValueError('Input encoding not available for this model')
    
    cpdef KBestTranslations translate(self,dinput,int k):
        raise ValueError('Direct translation not available for this model')

    cdef KShortestPaths decode_input(self,int[:] decode_input,int k):
        raise ValueError('Direct decoding input not available for this model')

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

        ## dump the current model 
        dtime = time.time()
        self.logger.info('Backing up the current model...')
        self.backup(config.dir)

        ## split the dataset up into n jobs and run the jobs 
        self.logger.info('Now setting up the jobs infrastructure...')
        it = time.time()
        _,rsize = setup_jobs(config)
        self.logger.info('Copied and setup up in %s seconds' % str(time.time()-it))

        ## score the joined together item
        merge = os.path.join(config.dir,"merged_ranks.txt")
        with open(merge) as mc: flen = sum([1 for i in mc])
        self.logger.info('file length: %d, rank_size=%d' % (flen,rsize))

        try: 
            #storage = RankStorage.load_from_file(merge,flen,rsize)
            #storage = RankStorage.load_from_file(merge,flen,config.k)
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
        
cdef class PolyglotConcurrentDecoder(ConcurrentWordModelDecoder):
    """Concurrent decoder model for polyglot models"""

    def __init__(self,model,graph,edge_labels,edge_map,lang_starts):
        """ 

        :param model: the underlying (word) translation model 
        :param edge_labels: word labels for edges in the graph 
        :param edge_map: point from edges to positions in the symbol table 
        :param lang_starts: the starting positions of difference languages 
        """
        self.model = <SymmetricWordModel>model
        self.graph = graph
        self.edge_labels = edge_labels
        self.edge_map = edge_map
        self.langs = lang_starts

    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset asynchronously by splitting the data into n pieces

        :param config: the main configuration, which specifies the data location 
        and the number of jobs to run.
        """
        self._setup_jobs(config)

    @classmethod
    def from_config(cls,config):
        """Loads a dag decoder from configuration 

        :param configuration: the target configuration 
        :type: zubr.util.config.ConfigAttrs
        :returns: a dag decoder instance 
        """
        ## setup the underlying model
        model = SymmetricWordModel.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels,langs = load_graph(config,model.ftoe.flex,poly=True)
        graph = WordGraph.from_adj(edges,spans,size)

        return cls(model,graph,smap,edge_labels,langs)

    def score_ranks(self,wdir,k,ranks,config=None):
        """Score the ranks according to the type of decoder 

        :param ranks: the formatted rank storage item 
        """
        cdef SymmetricWordModel model = <SymmetricWordModel>self.model
        cdef dict edict = model.ftoe.elex
        cdef dict fdict = model.ftoe.flex

        ## this should be fixed
        self.logger.info('Scoring the new rank list')
        stuff  = get_decoder_data(config,fdict,edict,ttype=config.eval_set,poly=True)
        ranks.compute_poly_score(wdir,stuff[-1],stuff[2],k,dtype='baseline')

    ## too much redundancy!!!
    def backup(self,wdir):
        """Back up a graph decoder model

        :param wdir: the working directory where to place model 
        :rtype: None 
        """
        stime = time.time()
        decoder_path = os.path.join(wdir,"graph_decoder")
        if os.path.isfile(decoder_path) or os.path.isdir(decoder_path):
            self.logger.info('Already backed up, skipping...')
            return 

        os.mkdir(decoder_path) 
        ## back up the underlying translation model 
        self.model.backup(wdir)
        ## back up the graph model 
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_path,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_path,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump({"edges":self.edge_map,"langs":self.langs},info)

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))
        
    load_backup = PolyglotWordDecoder.load_backup
        
    def __reduce__(self):
        ## pickle implementation
        return PolyglotWordDecoder,(self.model,self.graph,self.edge_labels,self.edge_map,self.langs)

cdef class WordGraphConcurrentDecoder(ConcurrentWordModelDecoder):
    """Concurrent decoder model for ordinary word models"""

    def __init__(self,model,graph,edge_labels,edge_map):
        """Create a word graph decoder instance 


        :param model: the underlying translation model 
        :param graph: the underlying word graph 
        :param symbols: the underlying symbol/path table 
        :param edge_map: point from edges to positions in symbol table 
        :param oov: the out of vocabular map
        """
        self.model       = <SymmetricWordModel>model
        self.graph       = graph
        self.edge_labels = edge_labels
        self.edge_map    = edge_map

    cpdef int decode_data(self,object config) except -1:
        """Decode a dataset asynchronously by splitting the data into n pieces

        :param config: the main configuration, which specifies the data location 
        and the number of jobs to run.
        """
        self._setup_jobs(config)

    @classmethod
    def from_config(cls,config):
        """Loads a dag decoder from configuration 

        :param configuration: the target configuration 
        :type: zubr.util.config.ConfigAttrs
        :returns: a dag decoder instance 
        """
        ## setup the underlying model
        model = SymmetricWordModel.from_config(config)

        ## setup the graph
        edges,spans,size,smap,edge_labels = load_graph(config,model.ftoe.flex)
        graph = WordGraph.from_adj(edges,spans,size)
        
        return cls(model,graph,smap,edge_labels)

    def score_ranks(self,wdir,k,ranks,config=None):
        """Score the ranks according to the type of decoder 

        :param ranks: the formatted rank storage item 
        """
        ranks.compute_mono_score(wdir,k,'baseline')

    def backup(self,wdir):
        """Back up a graph decoder model

        :param wdir: the working directory where to place model 
        :rtype: None 
        """
        stime = time.time()
        decoder_path = os.path.join(wdir,"graph_decoder")
        if os.path.isfile(decoder_path) or os.path.isdir(decoder_path):
            self.logger.info('Already backed up, skipping...')
            return 

        os.mkdir(decoder_path) 
        ## back up the underlying translation model 
        self.model.backup(wdir)
        ## back up the graph model 
        self.graph.backup(wdir)

        ## back up the edge labels
        fout = os.path.join(decoder_path,"edge_labels")
        np.savez_compressed(fout,self.edge_labels)

        ## back up the
        dinfo = os.path.join(decoder_path,"decoder_info.p")
        with gzip.open(dinfo,'wb') as info:
            pickle.dump(self.edge_map,info)

        ## log the time 
        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))
        
    load_backup = WordGraphDecoder.load_backup

    def __reduce__(self):
        ## pickle implementation 
        return WordGraphDecoder,(self.model,self.graph,self.edge_labels,self.edge_map)

## auxiliary classes
    
cdef class SequencePath(Path):
    """A kind of directed path specialized to the word decoding problem"""

    def __init__(self,np.ndarray path,
                     np.ndarray eseq,
                     np.ndarray useq,
                     np.ndarray node_scores,
                     double score,int size
        ):
        """Creates a sequence path instance 

        :param path: the actual path 
        :param score: the path score (-log probability of sequence)
        :param sscores: the sequence scores at each point 
        :param size: the length of the path
        """
        self.seq  = path
        self.eseq = eseq
        self.score = score
        self.size = size
        self.node_scores = node_scores
        self._translation = useq

    property encoding:
        """The underlying path sequence"""

        def __get__(self):
            """Return the raw, numpy sequence information 

            :rtype: np.ndarray
            """
            cdef np.ndarray[ndim=1,dtype=np.int32_t] eseq = self.eseq
            return eseq


    property null_encoding:
        """The translation encoding with null sumbol in front"""

        def __get__(self):
            """Return the raw, numpy sequence information with null symbol in front

            :rtype: np.ndarray
            """
            cdef np.ndarray[ndim=1,dtype=np.int32_t] eseq = self.eseq
            return np.insert(eseq,0,0)

    property translation:
        """The underlying path sequence"""

        def __get__(self):
            """Return the raw, numpy sequence information 

            :rtype: np.ndarray
            """
            cdef np.ndarray trans = self._translation
            return trans

    property path:
        """The underlying path sequence"""

        def __get__(self):
            """Return the raw, numpy sequence information 

            :rtype: np.ndarray
            """
            cdef np.ndarray seq = self.seq
            cdef double cost = self.score
            return (seq,cost)

    property invalid:
        """Path if the path is valid (i.e., final node has a normal (non-inf) score)"""

        def __get__(self):
            """Returns whether teh path is valid 

            :rtype: bool 
            """
            cdef double score = self.score
            return isinf(score)

    property translation_string:
        """Return only the unicode string"""
        
        def __get__(self):
            """Return the raw, numpy sequence information 

            :rtype: np.ndarray
            """
            cdef np.ndarray trans = self._translation
            return ' '.join(trans)
    

cdef class SimpleSequencePath(SequencePath):
    """Simplified sequence path for the hoffman implementation """

    def __init__(self,path,useq,size,eseq=np.empty(0,dtype=np.int32),score=1.0):
        """Creates a sequence path instance 

        :param path: the graph path sequence 
        :param eseq: the encoded sequence for translation model 
        :param useq: the unicode sequence 
        :param score: the score of the path 
        :param size: the size of the path sequence 
        """
        self.seq = path
        self.eseq = eseq
        self.score = score
        self._translation = useq
        self.size = size

    property invalid:
        def __get__(self):
            """In these sequence paths, all resulting paths are valid"""
            return False
        
cdef class SequenceBuffer:

    """Cython/c levle buffer for storing candidate paths. Repeated from Graph.pyx"""

    
    def __init__(self):
        """Initialized a path buffer instance 

        """
        self.seen = set()
        self.sortedpaths = []

    cdef void push(self,double cost, SequencePath path):
        """Push a candidate path onto heap

        Note: it assumes that they are new, unseen sequences 
        (this follows from implementing ``Lawler's`` trick)

        :param cost: the 
        :rtype: None
        """
        cdef list paths = self.sortedpaths
        heappush(paths,(cost,path))

    def __iter__(self):
        return iter(self.sortedpaths)
        
    cdef SequencePath pop(self):
        """Return the top item with lowest score

        :returns: a path instance 
        """
        cdef list paths = self.sortedpaths
        cdef double cost
        cdef SequencePath popped

        cost,popped = heappop(paths)
        return <SequencePath>popped

    cdef np.ndarray kbest(self,int k):
        """Return the top k sequences

        :param k: the number k to return 
        """
        cdef list paths = self.sortedpaths
        return np.array([z[1] for z in nsmallest(k,paths)],dtype=np.object)

    cdef bint empty(self):
        """Returns if buffer is empty 

        :rtype: bool
        """
        cdef list paths = self.sortedpaths
        return <bint>(not paths)
    
    def __len__(self):
        ## number of candidates 
        cdef list paths = self.sortedpaths
        return <int>len(paths)

    def __bool__(self):
        cdef list paths = self.sortedpaths
        return <bint>(paths == [])


## a reimplmentaiton of the k shortest paths class from graph

cdef class KBestTranslations:
    """Representing the k-best translations (paths) from word graph"""

    def __init__(self,einput,uinput,A,k):
        """Create a kbest translations instance 

        :param einput: the encoded englis input 
        :param uinput: the unicode string 
        :param A: the list of best translations/paths 
        :param k: the number of translations found 
        """
        self._einput = einput
        self._uinput = uinput
        self._k = k
        self._A = A[:k]

    def __iter__(self):
        ## iteration
        cdef np.ndarray A = self._A
        cdef int i,asize = A.shape[0]
        for i in range(asize):
            yield A[i]

    property text:
        """The input text representation"""
        
        def __get__(self):
            """Returns the number of items found"""
            cdef unicode english = self._uinput
            return english

    property encoding:
        """The input text representation"""
        
        def __get__(self):
            """Returns the number of items found"""
            cdef np.ndarray[ndim=1,dtype=np.int32_t] encode = self._einput
            return encode
        
    property size:
        """The number of items found"""
        
        def __get__(self):
            """Returns the number of items found"""
            cdef int k = self._k
            return k

    property paths:
        """The list of current paths"""

        def __get__(self):
            """Return the current paths 

            :rtype: np.ndarray
            """
            cdef np.ndarray paths = self._A
            return paths
        
DECODERS = {
    "wordgraph"     : WordGraphDecoder,
    "polyglot"      : PolyglotWordDecoder,
    "executable"    : ExecutableGraphDecoder,
    "pexecutable"   : ExecutablePolyGraphDecoder,
    "con_wordgraph" : WordGraphConcurrentDecoder,
    "con_polyglot"  : PolyglotConcurrentDecoder,
}

cpdef Decoder(str decoder_type):
    """Graph decoder factor

    :param decoder_type: the desired type of graph decoder 
    :type decoder_type: basestring 
    :returns: Graph Decoder class 
    :raises: ValueError
    """
    dtype = decoder_type.lower()
    if dtype not in DECODERS:
        raise ValueError('Uknown decoder type: %s' % decoder_type)
    return DECODERS[dtype]


## C level functions

cdef unicode to_unicode(s):
    if isinstance(s,bytes):
        return (<bytes>s).decode('utf-8')
    return s

cdef int pexecute_dataset(SymmetricWordModel model,
                            ExecutableModel executor,
                            np.ndarray dataset,
                            np.ndarray dataset_orig,
                            np.ndarray langs,
                            np.ndarray grep,
                            int k,
                            WordGraph graph,
                            int[:] edge_labels,
                            dict emap,
                            dict rmap,
                            int[:] gid,
                            object logger,
                            str etype,
                            directory=None
                              ) except -1:
    """Evaluate a dataset with an executable graph decoding model
    
    """
    cdef DirectedAdj adj = graph.adj
    cdef int gsize = graph.num_nodes

    ## dataset info
    cdef int i,size = dataset.shape[0]
    cdef unicode esentence,utranslation,gtranslation
    
    ## output paths 
    cdef KBestTranslations paths
    cdef SequencePath path

    cdef int identifier,num
    cdef double st

    cdef int rsize = len(rmap)
    cdef RankStorage storage = RankStorage.load_empty(size,k+1)
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos
    cdef dict other_gold = storage.other_gold

    cdef bint gold_found,matching_den
    cdef int gnoi = 0

    ## model 
    cdef WordModel aligner = <WordModel>model.ftoe
    cdef double[:,:] table = aligner.model_table()
    
    logger.info('Decoding polyglot dataset...')

    st = time.time()

    for i in range(size):
        other_gold[i] = set()

        ## check if gold "answer" is found
        gold_found = False
        esentence = np.unicode(dataset_orig[i])
        gtranslation = np.unicode(grep[gid[i]])
        
        try: 
            paths = shortest_path(adj,edge_labels,
                                    esentence,
                                    gsize,
                                    k+1,
                                    0,
                                    dataset[i],
                                    #model,
                                    aligner,
                                    table,
                                    emap)
        except Exception,e:
            logger.error(e,exc_info=True)

        ## go through each output an execute
        for num,path in enumerate(paths):
            utranslation = path.translation_string
            ## add to rmap if not there already 
            if utranslation not in rmap:
                rmap[utranslation] = len(rmap)
                
            identifier = rmap[utranslation]
            ranks[i][num] = identifier
            matching_den = executor.evaluate(gtranslation,utranslation)
            if not gold_found and matching_den:
                gold_found  = True
                gold_pos[i] = num

            ## keep a record of other items that evaluate to true
            elif matching_den:
                other_gold[i].add(num)

        if not gold_found:
            ## maybe log this?
            #gold_pos[i] = (rsize - 1)
            gold_pos[i] = k
            ranks[i][k] = gid[i]
            gnoi += 1

    # # log time information 
    logger.info('decoded and scored %d sentences in %s seconds (%d not in beam), other_gold=%s' %\
                    (size,str(time.time()-st),gnoi,str(other_gold != {})))

    # ## score if desired
    if directory:
        storage.compute_poly_score(directory,langs,rmap,k,dtype='baseline',exc=True)
        storage.backup(directory,name=etype)

cdef int execute_dataset(SymmetricWordModel model,
                            ExecutableModel executor,
                            np.ndarray dataset,
                            np.ndarray dataset_orig,
                            np.ndarray grep,
                            int k,
                            WordGraph graph,
                            int[:] edge_labels,
                            dict emap,
                            dict rmap,
                            int[:] gid,
                            object logger,
                            etype,
                            directory=None) except -1:
    """Evaluate a dataset with an executable graph decoding model
    
    """
    ## graph properties 
    cdef DirectedAdj adj = graph.adj
    cdef int gsize = graph.num_nodes

    ## dataset info
    cdef int i,size = dataset.shape[0]
    cdef unicode esentence,utranslation,gtranslation
    
    ## output paths 
    cdef KBestTranslations paths
    cdef SequencePath path

    cdef int identifier,num
    cdef double st

    ## rank object
    cdef RankStorage storage = RankStorage.load_empty(size,k+1)
    #cdef RankStorage storage = RankStorage.load_empty(size,k)
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos
    cdef dict other_gold = storage.other_gold

    cdef bint gold_found,matching_den
    cdef int gnoi = 0

    ## model used for scoring 
    cdef WordModel aligner = <WordModel>model.ftoe
    cdef double[:,:] table = aligner.model_table()
    
    logger.info('Decoding dataset, dsize=%d, k=%d' % (len(rmap),k))
    st = time.time()

    ## go through each data point

    for i in range(size):
        other_gold[i] = set()

        ## check if gold "answer" is found
        gold_found = False
        esentence = np.unicode(dataset_orig[i])
        gtranslation = np.unicode(grep[gid[i]])

        try: 
            paths = shortest_path(adj,edge_labels,
                                    esentence,
                                    gsize,
                                    k+1,
                                    0,
                                    dataset[i],
                                    #model,
                                    aligner,
                                    table,
                                    emap)
        except Exception,e:
            logger.error(e,exc_info=True)

        ## go through each output an execute
        for num,path in enumerate(paths):
            utranslation = path.translation_string
            ## add to rmap if not there already 
            if utranslation not in rmap:
                rmap[utranslation] = len(rmap)
                
            identifier = rmap[utranslation]
            ranks[i][num] = identifier
            matching_den = executor.evaluate(gtranslation,utranslation)
            if not gold_found and matching_den:
                gold_found  = True
                gold_pos[i] = num

            ## keep a record of other items that evaluate to true
            elif matching_den:
                other_gold[i].add(num)

        if not gold_found:
            ## maybe log this?
            gold_pos[i] = k
            ranks[i][k] = gid[i]
            gnoi += 1

    if directory:
        #storage.compute_score(directory,'baseline')
        ## print out the new rank link
        storage.compute_mono_score(directory,k,'baseline')
        storage.backup(directory,name=etype)

    ## log the information 
    logger.info('Decoded and scored %d sentences in %s seconds (%d not in beam),rmap=%d, other_gold=%s' %\
                    (size,str(time.time()-st),gnoi,len(rmap),str(other_gold != {})))
                    
                    

cdef int score_poly(SymmetricWordModel model,
                np.ndarray dataset,
                np.ndarray dataset_orig,
                np.ndarray langs,
                int k,
                WordGraph graph,
                int[:] edge_labels,
                dict emap,
                dict rmap,
                dict lang_map,
                int[:] gid,
                bint spec_lang,
                object logger,
                directory=None,
                set blocks=set()
                ) except -1:
    """Score a polyglot dataset

    :param model: the translation model 
    :param dataset: the dataset to decode
    :param dataset_orig: the original representations of the dataset 
    :param langs: the language identifier for this dataset
    :param k: the number of best sequences (or shortest paths) to return 
    :param graph: the word graph 
    :param edge_labels: the label identifiers for graph arcs 
    :param emap: the labels to unicode text 
    :param rmap: the rank list map (components to identifiers in global list)
    :param langs: the language starting points 
    :param gid: the gid identifiers 
    :param logger: the decoder instance logger 
    :param directory: the working directory (to back up output)
    """
    ## graph properties 
    cdef DirectedAdj adj = graph.adj
    cdef int gsize = graph.num_nodes

    ## dataset info
    cdef int i,size = dataset.shape[0]
    cdef unicode esentence,utranslation
    
    ## output paths 
    cdef KBestTranslations paths
    cdef SequencePath path

    cdef int identifier,num
    cdef double st

    ## language counts
    cdef int rsize = len(rmap)
    cdef RankStorage storage = RankStorage.load_empty(size,k+1)
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos

    ## models
    cdef WordModel aligner = <WordModel>model.ftoe
    cdef double[:,:] table = aligner.model_table()
    
    ##
    cdef bint gold_found
    cdef int gnoi = 0
    cdef int source
    cdef set repeats
    
    logger.info('Decoding polyglot dataset, set_lang=%s ...' % str(spec_lang))

    ## start the counter 
    st = time.time()

    for i in range(size):
        if ((i+1) % 1000) == 0: logger.info('parsing number: %d' % i)
        
        ## indicates whether gold item is found 
        gold_found = False 
        esentence = np.unicode(dataset_orig[i])

        ## where to start the search? 
        source = 0 if not spec_lang else lang_map[langs[i]][1]
        
        ## find the best paths
        try:
            paths = shortest_path(adj,edge_labels,
                                    esentence,
                                    gsize,
                                    k+1,
                                    source,
                                    dataset[i],
                                    #model,
                                    aligner,
                                    table,
                                    emap,
                                    blocks=blocks
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

            ## unknown output? 
            if identifier == -1:
                logger.warning('Unknown output: %s for input %d' % (utranslation,i))

        if not gold_found:
            #gold_pos[i] = k
            gold_pos[i] = (rsize-1)
            #ranks[i][k] = gid[i]
            gnoi += 1
            
    # # log time information 
    logger.info('decoded and scored %d sentences in %s seconds (%d not in beam)' %\
                    (size,str(time.time()-st),gnoi))

    # ## score if desired
    if directory:
        storage.compute_poly_score(directory,langs,rmap,k,dtype='baseline')
    

@boundscheck(False)
@cdivision(True)
cdef void score_dataset(SymmetricWordModel model,#double[:,:] table,
                        np.ndarray dataset,
                        np.ndarray dataset_orig,
                        int start_node,
                        int k,
                        WordGraph graph,
                        int[:] edge_labels,
                        dict emap,
                        dict rmap,
                        int[:] gid,
                        object logger,
                        etype,
                        directory=None
                        ):
    """Decode a dataset and score 

    :param table: the translation model parameters 
    :param dataset: the input dataset 
    :param dataset_orig: the original dataset 
    :param start_node: where to start the search in the graph 
    :param k: the number of paths to find in graph (maximum) 
    :param graph: the underlying graph 
    :param edge_labels: the graph edge labels 
    :param emap: the edge labels for building original representations 
    :param rmap: the map from outputs to identifiers 
    :param gid: the gold identifiers 
    :param logger: the graph instance logger 
    :param hoffman: whether to use the hoffman shit 
    """
    ## graph properties 
    cdef DirectedAdj adj = graph.adj
    cdef int gsize = graph.num_nodes
    
    ## dataset info
    cdef int i,size = dataset.shape[0]
    cdef unicode esentence,utranslation
    
    ## output paths 
    cdef KBestTranslations paths
    cdef SequencePath path

    cdef int identifier,num
    cdef double st = time.time()

    ## rank score object
    cdef int rsize = len(rmap)
    cdef RankStorage storage = RankStorage.load_empty(size,rsize)
    cdef int[:,:] ranks = storage.ranks
    cdef int[:] gold_pos = storage.gold_pos

    ## 
    cdef bint gold_found
    cdef int gnoi = 0
    cdef set repeats

    ## model used for scoring 
    cdef WordModel aligner = <WordModel>model.ftoe
    cdef double[:,:] table = aligner.model_table()
    
    logger.info('Decoding dataset, rlen=%d' % rsize)

    ## go through data and decode
    for i in range(size):
        
        ## check if the gold item is found 
        gold_found = False
        esentence = np.unicode(dataset_orig[i])
        
        #if not hoffman: 
        paths = shortest_path(adj,edge_labels,
                                  esentence,
                                  gsize,
                                  k+1,
                                  start_node,
                                  dataset[i],
                                  #model,
                                  aligner,
                                  table,
                                  emap)
        # else:
        #     paths = hoffman_shortest_path(adj,edge_labels,
        #                                 esentence,
        #                                 gsize,
        #                                 k+1,
        #                                 start_node,
        #                                 dataset[i],
        #                                 #model,
        #                                 aligner,
        #                                 table,
        #                                 emap)
            

        ## go through the difference paths
        repeats = set()
        
        for num,path in enumerate(paths):
            
            ## the unicode translation string and identifier in rank
            utranslation = path.translation_string
            identifier = rmap.get(utranslation,-1)
            ranks[i][num] = identifier

            ## check for repeats:
            if identifier in repeats:
                logger.info('found a repeat: %s' % utranslation)
            repeats.add(identifier)
            
            ## gold position? 
            if identifier == gid[i]:
                gold_found  = True
                gold_pos[i] = num

            ## unknown output? 
            if identifier == -1:
                logger.warning('Unknown output: %s for input %d' % (utranslation,i))

        if not gold_found:
            gold_pos[i] = (rsize - 1)
            ## do not artifically put this here 
            ranks[i][rsize-1] = gid[i]
            gnoi += 1
            
    ## log the time 
    logger.info('decoded and scored %d sentences in %s seconds (%d not in beam)' %\
                    (size,str(time.time()-st),gnoi))

    ## evaluate (if necessary)
    if directory:
        #storage.compute_score(directory,'baseline')
        storage.compute_mono_score(directory,k,'baseline')
        storage.backup(directory,name=etype)

@boundscheck(False)
@cdivision(True)
cdef KBestTranslations hoffman_shortest_path(DirectedAdj adj,int[:] labels, # graph components
                                      unicode uinput,
                                      int size, ## the number of edges 
                                      int k, ## the size of the
                                      int starting_point,
                                      np.ndarray[ndim=1,dtype=np.int32_t] english,
                                      WordModel aligner,
                                      double[:,:] table,
                                      dict emap,
                                      set blocks=set(), ## global blocks 
                                      ):
    """A modified shortest path algorithm for finding best paths in word graphs given input

    :param adj: the adjacency list 
    :param size: the number of nodes 
    :param k: the number of desired paths 
    :param english: the english or nl input
    :param table: the probability table 
    :param emap: the map for looking up edge unicode labels
    """
    ## graph components 
    cdef int[:] edges   = adj.edges
    cdef int[:,:] spans = adj.node_spans

    ## aligner
    cdef Alignment model_alignment

    ## k best list 
    cdef np.ndarray sortest_k_best,A = np.ndarray((k,),dtype=np.object)
    cdef SequencePath top_candidate,previous,recent_path
    cdef SimpleSequencePath spur_path,new_path
    cdef SequenceBuffer B = SequenceBuffer()
    cdef SequenceBuffer AS = SequenceBuffer()

    ## the different sequences involved -
    cdef np.ndarray[ndim=1,dtype=np.int32_t] prev_seq, spur_seq,total_path,root_path,other_path,prev_eseq
    cdef np.ndarray[ndim=1,dtype=np.int32_t] total_eseq,spur_eseq
    cdef np.ndarray prev_trans,spur_trans,total_trans

    cdef dict spurs_seen = <dict>defaultdict(set)
    ## equality lookup
    cdef int[:] equal_path = np.zeros((k,),dtype=np.int32)
    
    ## input information
    cdef isize = english.shape[0]

    cdef int current_k,i,j,prev_size
    cdef int root_len
    cdef int root_outgoing,block_len
    cdef double total_score,top_score,prev_num

    ## block information
    cdef set root_block,observed_spurs #,ignored_nodes,ignored_edges
    cdef empty_scores = np.zeros((isize,),dtype='d')
    ## cdef
    cdef dict edge_priority = {}
    cdef int[:] node_list
    cdef int next_node
    cdef double new_score
    cdef set ignore_nodes,ignore_edges
    cdef unicode o

    ## elex
    cdef dict flex = aligner.flex
    
    ### static k-shortest path information

    previous = trans_edge_scores(0,edge_priority,empty_scores,0,edges,spans,labels,
                                     english,isize,size,table,emap,True)
    A[0] = previous
    current_k = 1

    ## model score for first-best 
    model_alignment = aligner._align(previous.null_encoding,english)
    top_score = -log(model_alignment.prob)

    while True:
        if current_k >= k: break
        ## reset equal path
        equal_path[:] = 0

        ## initialize what to ignore
        ignore_nodes = set()
        ignore_edges = set()

        ## add global blocks 
        ignore_edges = ignore_edges.union(blocks)

        ## previous size and actual sequence
        prev_size  = previous.size
        prev_seq   = previous.seq
        prev_eseq  = previous.eseq
        prev_trans = previous.translation
        prev_num   = previous.score

        ## push previous onto separate heap
        AS.push(top_score, previous)

        for i in range(1,prev_size):
            root_path = prev_seq[:i]
            root_len = root_path.shape[0]
            root_block = set()

            ## number of outgoing
            root_outgoing = (spans[root_path[-1]][-1]+1)-spans[root_path[-1]][0]

            ## sequence score at this node
        
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

            ignore_nodes.add(root_path[-1])
            block_len = len(root_block)

            ## check if all edges have been exhausted    
            if block_len == root_outgoing: continue
                
            ## check if restriction have already been looked at (Lawler's rule)
            observed_spurs = spurs_seen[tuple(root_path)]
            if block_len <= len(observed_spurs): continue
            observed_spurs.update(root_block)

            ## find the next edge from this spur
            node_list = edge_priority[root_path[-1]]

            ## get the next best path from here
            spur_path = next_best(root_path[-1],
                                      node_list,
                                      edge_priority,
                                      labels,
                                      size,
                                      emap,
                                      200,
                                      ignore_edges,
                                      len(node_list)
                                      )

            ## sput path and score 
            spur_seq   = spur_path.seq

            ## eseq and translation 
            spur_eseq = spur_path.eseq
            spur_trans = spur_path.translation
            total_path = np.concatenate((root_path,spur_seq[1:]),axis=0)

            ## concacenate encoded sequence            
            total_trans = np.concatenate((prev_trans[:i-1],spur_trans))
            total_eseq = np.array([flex.get(o,-1) for o in total_trans],dtype=np.int32)

            ## new cnadidate path 
            new_path = SimpleSequencePath(total_path,
                                        total_trans,
                                        total_path.shape[0],
                                        total_eseq)
            
            ## get the correct probability 
            model_alignment = aligner._align(new_path.null_encoding,english)
            model_score = -log(model_alignment.prob)
            new_path.score = model_score
            B.push(model_score,new_path)

        ## empty candidate list
        if B.empty(): break

        ## get the top candidate 
        top_candidate = B.pop()
        top_score = top_candidate.score
        A[current_k] = top_candidate
        current_k += 1
        previous = top_candidate

    ## after
    sortest_k_best = AS.kbest(current_k)
    return KBestTranslations(english,uinput,sortest_k_best,current_k) 

        
@boundscheck(False)
@cdivision(True)
cdef KBestTranslations shortest_path(DirectedAdj adj,int[:] labels, # graph components
                                      unicode uinput,
                                      int size, ## the number of edges 
                                      int k, ## the size of the
                                      int starting_point,
                                      np.ndarray[ndim=1,dtype=np.int32_t] english,
                                      #double[:,:] table, ## the translation model parameters
                                      #SymmetricWordModel model,
                                      WordModel aligner,
                                      double[:,:] table,
                                      dict emap,
                                      set blocks=set(), ## global blocks 
                                      ):
    """A modified shortest path algorithm for finding best paths in word graphs given input

    :param adj: the adjacency list 
    :param size: the number of nodes 
    :param k: the number of desired paths 
    :param english: the english or nl input
    :param table: the probability table 
    :param emap: the map for looking up edge unicode labels
    """
    ## graph components 
    cdef int[:] edges   = adj.edges
    cdef int[:,:] spans = adj.node_spans

    ## aligner 
    #cdef WordModel aligner = <WordModel>model.ftoe
    #cdef double[:,:] table = aligner.table
    #cdef double[:,:] table = aligner.model_table()
    cdef Alignment model_alignment
    
    ## k best list 
    cdef np.ndarray sortest_k_best,A = np.ndarray((k,),dtype=np.object)
    cdef SequencePath top_candidate,previous,spur_path,new_path,recent_path
    cdef SequenceBuffer B = SequenceBuffer()
    cdef SequenceBuffer AS = SequenceBuffer()

    ## the different sequences involved -
    cdef np.ndarray[ndim=1,dtype=np.int32_t]  prev_seq, spur_seq,total_path,root_path,other_path,prev_eseq
    cdef np.ndarray[ndim=1,dtype=np.int32_t]  total_eseq,spur_eseq
    cdef np.ndarray[ndim=1,dtype=np.double_t] root_score
    cdef np.ndarray[ndim=2,dtype=np.double_t] prev_score,spur_score,total_node_score
    cdef np.ndarray prev_trans,spur_trans,total_trans


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

    ## first shortest path
    previous = best_sequence_path(starting_point,empty_scores,0,edges,spans,labels,
                                      english,isize,size,table,emap,True)
    A[0] = previous
    current_k = 1

    while True:

        if current_k >= k: break
        ## reset equal path
        equal_path[:] = 0

        ## initialize what to ignore
        ignore_nodes = set()
        ignore_edges = set()
        ## add global blocks 
        ignore_edges = ignore_edges.union(blocks)
        ignore_nodes = ignore_nodes.union(blocks)

        ## previous size and actual sequence
        prev_size  = previous.size
        prev_seq   = previous.seq
        prev_score = previous.node_scores
        prev_eseq  = previous.eseq
        prev_trans = previous.translation
        prev_num   = previous.score

        ## push previous onto separate heap

        ## align with ordinary model to ensure that score is correct
        model_alignment = aligner._align(previous.null_encoding,english)
        #AS.push(-model_alignment.prob,previous)
        AS.push(-log(model_alignment.prob),previous)

        for i in range(1,prev_size):
            root_path = prev_seq[:i]
            root_len = root_path.shape[0]

            ## root score
            root_score = prev_score[i-1]

            ## number of outgoing
            root_outgoing = (spans[root_path[-1]][-1]+1)-spans[root_path[-1]][0]
            root_block = set()
            
            ## sequence score at this node
        
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

            ignore_nodes.add(root_path[-1])
            block_len = len(root_block)
            
            ## check if all edges have been exhausted    
            if block_len == root_outgoing: continue
                
            ## check if restriction have already been looked at (Lawler's rule)
            observed_spurs = spurs_seen[tuple(root_path)]
            if block_len <= len(observed_spurs): continue
            observed_spurs.update(root_block)

            ## do next best path search
            spur_path = best_sequence_path(root_path[-1],root_score,i,
                                            edges,spans,labels,
                                            english,isize,size,table,
                                            emap,False,
                                            ignored_edges=ignore_edges,
                                            ignored_nodes=ignore_nodes)
            
            ## sput path and score 
            spur_seq   = spur_path.seq
            spur_score = spur_path.node_scores

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

            ## new cnadidate path 
            new_path = SequencePath(total_path,
                                        total_eseq,
                                        total_trans,
                                        total_node_score,
                                        total_score,
                                        total_path.shape[0])

            B.push(total_score,new_path)

        ## empty candidate list
        if B.empty(): break

        ## get the top candidate
        top_candidate = B.pop()
        top_score = top_candidate.score

        ## k-best list for best path search
        A[current_k] = top_candidate

        current_k += 1
        previous = top_candidate

    sortest_k_best = AS.kbest(current_k)
    #return KBestTranslations(english,uinput,A,current_k)
    return KBestTranslations(english,uinput,sortest_k_best,current_k)


## DAG best path

@wraparound(False)
@boundscheck(False)
@cdivision(True)
cdef SimpleSequencePath next_best(int source,
                                int[:] node_list,
                                dict edge_priority,
                                #int next_edge,
                                int[:] labels,
                                int size,
                                dict emap,
                                int max_size,
                                set ignore_edges,
                                int source_len
                                ):
    """Computes the next best using blocks and the k-best edge information 

    :param source: the source node 
    :param next_edge: the next edge to pursue from source 
    :param edge_priority: the sorted list of edges to use 
    :param labels: the edge labels 
    :param emap: the map for looking up edge unicode labels
    :param max_size: the maximum size output string 
    """
    cdef int i,j,current_len,edge_len
    cdef int *node_seq = <int *>malloc(max_size*sizeof(int))
    cdef np.ndarray[ndim=1,dtype=np.int32_t] end_seq
    cdef np.ndarray unicode_seq
    cdef int next_edge,node

    try:

        with nogil: 
            for i in range(max_size): node_seq[i] = 0
            ## add first label
            node_seq[0] = source
            current_len = 1

            ### find the next best edge 
            for j in range(source_len):
                node = node_list[j]
                with gil:
                    if (source,node) in ignore_edges: continue
                next_edge = node
                break

            while True:
                if next_edge == (size-1): break
                node_seq[current_len] = next_edge
                with gil: node_list = edge_priority[next_edge]
                next_edge = node_list[0]
                current_len += 1

        # ## with the gil back on
        end_seq = np.array([i for i in node_seq[:current_len]],dtype=np.int32)
        unicode_seq = np.array([emap[(end_seq[i],end_seq[i+1])] for i in range(current_len-1)],dtype=np.object)
        
        return SimpleSequencePath(
            end_seq,
            unicode_seq,
            current_len
            )

    finally:
        free(node_seq) 
        
@wraparound(False)
@boundscheck(False)
@cdivision(True)
cdef SequencePath trans_edge_scores(int source,
                                    dict edge_scores,
                                    double[:] start_score,int start_len,
                                    int[:] edges,int[:,:] spans,
                                    int[:] labels, ## graph edge labels 
                                    int[:] einput,int esize, ## english input
                                    int size,
                                    double[:,:] table, ## translation parameters
                                    dict emap,
                                    bint first=False,
                                    ):
    """Computes the best path and information about the k-shortest positions

    """
    cdef int i,j,w,start,end,node
    cdef double cdef_float_infp = float('+inf')

    ## the resulting scores and sequences
    cdef np.ndarray[ndim=1,dtype=np.int32_t] end_seq
    cdef np.ndarray[ndim=2,dtype=np.double_t] fnode_scores
    cdef np.ndarray[ndim=1,dtype=np.int32_t] encoded_sequence
    cdef np.ndarray unicode_seq

    ## score information during search
    cdef double *d            = <double *>malloc(size*sizeof(double))
    cdef int *p               = <int *>malloc(size*sizeof(int))
    cdef double *node_scores  = <double *>malloc(size*sizeof(double))
    cdef int *out_seq         = <int *>malloc(size*sizeof(int))

    ## edge labels 
    cdef int *best_labels     = <int *>malloc(size*sizeof(int))
    cdef int *final_labels    = <int *>malloc(size*sizeof(int))

    ## best sequence length at each node 
    cdef double *nseq_len     = <double *>malloc(size*sizeof(double))
    
    ## sequence scores at each node 
    cdef double *best_seq    = <double *>malloc(size*esize*sizeof(double))
    cdef int *final_seq      = <int *>malloc(size*esize*sizeof(int))

    ## sequence lengths
    
    ## node sequence scores
    cdef int current,seq_len,xstart
    cdef double seq_score,table_score
    cdef int cword
    cdef list xitems

    ## divisor
    cdef double es = float(esize)
    cdef double ss = float(start_len)
    cdef tuple n

    try:
        with nogil:

            ## initialization
            for i in range(source,size):
                d[i]            = cdef_float_infp
                p[i]            = -1
                out_seq[i]      = -1
                node_scores[i]  = cdef_float_infp
                nseq_len[i]     = 0.0
                best_labels[i]  = 0
                final_seq[i]    = -1

                ### add edge scores 
                with gil: edge_scores[i] = []
                ## initialize word position scores at each node
                xstart = i*esize
                for j in range(esize):
                    best_seq[xstart+j] = 0.0

            ## initialize with null score if starting
            if first:

                ## compute null probabilities
                for j in range(esize):
                    if einput[j] == -1: continue
                    best_seq[(source*esize)+j] = table[0,einput[j]]
                    #best_seq[j] = table[0,einput[j]]

            ## source initiailization
            else:

                ## add starting scores
                for j in range(esize):
                    best_seq[(source*esize)+j] = start_score[j]

                ## add starting length
                if source == 0:
                    nseq_len[source] = 0
                else:
                    nseq_len[source] = ss


            ### go through now
            ## go through from sink to end
            for i in range(source,size-1):
                start = spans[i][0]
                end = spans[i][1]

                ## compute node sequence score
                if i > source and (p[i] == -1 or isinf(d[i])): continue
                    
                ## adjacencies
                for j in range(start,end+1):
                    node = edges[j]
                    seq_score = 1.0

                    ## score using the english side 
                    for w in range(esize):
                        
                        ## unknown e word
                        if einput[w] == -1: continue

                        ## component edge encodoing
                        cword = labels[j]
                        if cword == -1:
                            seq_score *=  best_seq[(i*esize)+w]
                        else:
                            table_score = table[cword,einput[w]]
                            seq_score *= table_score + best_seq[(i*esize)+w]

                    ## add the edge scores 
                    with gil:
                        #edge_scores[i][node] = -seq_score
                        #heappush(edge_scores[i],(-seq_score,node))
                        heappush(edge_scores[i],(-seq_score,node))

                    ## a terminating node? normalize score by size to get real probability
                    if node == size - 1:

                        ## normalization (add 1 to account for null word)
                        seq_score = seq_score/((nseq_len[i]+1)**es)

                        ## shorter path to end?
                        if d[node] > -seq_score:
                            d[node] = -seq_score
                            p[node] = i
                            nseq_len[node] = (nseq_len[i] + 1.0)
                            best_labels[node] = labels[j]

                    ## non terminating node?
                    elif (d[node] > -seq_score):

                        ## shorter path?
                        d[node] = -seq_score
                        p[node] = i
                        nseq_len[node] = (nseq_len[i] + 1.0)
                        best_labels[node] = labels[j]
                        
                        ## update counts for computing sequence score
                        for w in range(esize):
                            if einput[w] == -1: continue
                            ## component edge encodoing
                            cword = labels[j]
                            if cword == -1:
                                best_seq[(node*esize)+w] = best_seq[(i*esize)+w]
                            else:
                                best_seq[(node*esize)+w] = table[cword,einput[w]]+best_seq[(i*esize)+w]

                ## sort the edge scores
                with gil:
                  xitems = edge_scores[i]
                  edge_scores[i] = np.array([n[1] for n in xitems],dtype=np.int32)
                                                          
            out_seq[0] = size-1
            node_scores[0] = d[size-1]
            current = size-1
            seq_len = 1
            weight = d[current]
            final_seq[0] = best_labels[size-1]
            
            ## need to keep track of scores on each node
            while True:
                current = p[current]
                out_seq[seq_len] = current
                node_scores[seq_len] = d[current]
                final_seq[seq_len]   = best_labels[current]
                if current <= source: break
                seq_len += 1

         ## check if it is invalid
        ## compute resulting sequences 
        end_seq = np.array([i for i in out_seq[:seq_len+1]],dtype=np.int32)[::-1]

        fnode_scores = np.array(
            [np.array([best_seq[(i*esize)+j] for j in range(esize)],dtype='d') for i in end_seq],
            dtype='d'
        )

        ## only go to :seq_len because last symbol is a special graph token *END*
        encoded_sequence = np.array([i for i in final_seq[:seq_len]],dtype=np.int32)[::-1][:seq_len-1]
        ## unicode sequence
        unicode_seq =  np.array([emap[(end_seq[i],end_seq[i+1])] for i in range(seq_len-1)],dtype=np.object)

        return SequencePath(
            end_seq,
            encoded_sequence,
            unicode_seq,
            fnode_scores,
            weight,
            seq_len+1
        )

    finally:
        free(d)
        free(p)
        free(node_scores)
        free(out_seq)
        free(best_seq)
        free(nseq_len)
        free(best_labels)
        free(final_labels)
        free(final_seq)


@wraparound(False)
@boundscheck(False)
@cdivision(True)
cdef SequencePath best_sequence_path(int source,
                           double[:] start_score,int start_len,
                           int[:] edges,int[:,:] spans,
                           int[:] labels, ## graph edge labels 
                           int[:] einput,int esize, ## english input
                           int size,
                           double[:,:] table, ## translation parameters
                           dict emap,
                           bint first=False,
                           set ignored_edges=set(),## edges to ignore
                           set ignored_nodes=set(),## nodes to ignore
                           ):
    """Computes the best sequence path using translation model 

    :param source: the place to start in the graph 
    :param start_score: the previous scores
    :param start_len: the starting size of the sequence 
    :param edges: the graph edges 
    :param spans: pointers to the outgoing nodes for each starting node
    :param labels: the edge labels (encoded in terms of translation model rep)
    :param einput: the encoded english input
    :param esize: the size of the english input 
    :param size: the number of nodes in graph 
    :param table: the translation model parameters 
    :param emap: the edge label map to unicode representations
    :param first: whether this is the first best path being performed 
    :param ignored_edges: edges to ignore during search 
    :param ignored_nodes: nodes to ignore during search 

    note : it assumes that the graph is already topologically sorted! 
    It also assumes that there is a single end node!

    """
    cdef int i,j,w,start,end,node
    cdef double cdef_float_infp = float('+inf')

    ## the resulting scores and sequences
    cdef np.ndarray[ndim=1,dtype=np.int32_t] end_seq
    cdef np.ndarray[ndim=2,dtype=np.double_t] fnode_scores
    cdef np.ndarray[ndim=1,dtype=np.int32_t] encoded_sequence
    cdef np.ndarray unicode_seq

    ## score information during search
    cdef double *d            = <double *>malloc(size*sizeof(double))
    cdef int *p               = <int *>malloc(size*sizeof(int))
    cdef double *node_scores  = <double *>malloc(size*sizeof(double))
    cdef int *out_seq         = <int *>malloc(size*sizeof(int))

    ## edge labels 
    cdef int *best_labels     = <int *>malloc(size*sizeof(int))
    cdef int *final_labels    = <int *>malloc(size*sizeof(int))

    ## best sequence length at each node 
    cdef double *nseq_len     = <double *>malloc(size*sizeof(double))
    
    ## sequence scores at each node 
    cdef double *best_seq    = <double *>malloc(size*esize*sizeof(double))
    cdef int *final_seq      = <int *>malloc(size*esize*sizeof(int))
    
    ## sequence lengths
    
    ## node sequence scores
    cdef int current,seq_len,xstart
    cdef double seq_score,table_score
    cdef int cword

    ## divisor
    cdef double es = float(esize)
    cdef double ss = float(start_len)
    
    try:

        with nogil:

            ## initialization 
            #for i in range(size): ## change this to source -> size (rather than doing full loop)?
            for i in range(source,size):
                d[i]            = cdef_float_infp
                p[i]            = -1
                out_seq[i]      = -1
                node_scores[i]  = cdef_float_infp
                nseq_len[i]     = 0.0
                best_labels[i]  = 0
                final_seq[i]    = -1

                ## initialize word position scores at each node
                xstart = i*esize
                for j in range(esize):
                    best_seq[xstart+j] = 0.0

            ## initialize with null score if starting
            if first:

                ## compute null probabilities
                for j in range(esize):
                    if einput[j] == -1: continue
                    best_seq[(source*esize)+j] = table[0,einput[j]]
                    #best_seq[j] = table[0,einput[j]]

            ## source initiailization
            else:

                ## add starting scores
                for j in range(esize):
                    best_seq[(source*esize)+j] = start_score[j]

                ## add starting length
                if source == 0:
                    nseq_len[source] = 0
                else:
                    nseq_len[source] = ss

            ## go through from sink to end
            for i in range(source,size-1):
                start = spans[i][0]
                end = spans[i][1]

                ## compute node sequence score
                if i > source and (p[i] == -1 or isinf(d[i])): continue
                    
                ## adjacencies
                for j in range(start,end+1):
                    node = edges[j]

                    ## block nodes/edges 
                    with gil:
                        if node in ignored_nodes: continue
                        if (i,node) in ignored_edges: continue
                    ## move this above to prune out stuff 
                    ## compute node sequence score
                    #if i > source and (p[i] == -1 or isinf(d[i])): continue
                        
                    seq_score = 1.0

                    ## score 
                    for w in range(esize):
                        
                        ## unknown e word
                        if einput[w] == -1: continue

                        ## component edge encodoing
                        cword = labels[j]
                        if cword == -1:
                            seq_score *=  best_seq[(i*esize)+w]
                        else:
                            table_score = table[cword,einput[w]]
                            seq_score *= table_score + best_seq[(i*esize)+w]

                    ## a terminating node? normalize score by size to get real probability
                    if node == size - 1:

                        ## normalization (add 1 to account for null word)
                        seq_score = seq_score/((nseq_len[i]+1)**es)

                        ## shorter path to end?
                        if d[node] > -seq_score:
                            d[node] = -seq_score
                            p[node] = i
                            nseq_len[node] = (nseq_len[i] + 1.0)
                            best_labels[node] = labels[j]

                    ## non terminating node?
                    elif (d[node] > -seq_score):

                        ## shorter path?
                        d[node] = -seq_score
                        p[node] = i
                        nseq_len[node] = (nseq_len[i] + 1.0)
                        best_labels[node] = labels[j]
                        
                        ## update counts for computing sequence score
                        for w in range(esize):
                            if einput[w] == -1: continue
                            ## component edge encodoing
                            cword = labels[j]
                            if cword == -1:
                                best_seq[(node*esize)+w] = best_seq[(i*esize)+w]
                            else:
                                best_seq[(node*esize)+w] = table[cword,einput[w]]+best_seq[(i*esize)+w]

            out_seq[0] = size-1
            node_scores[0] = d[size-1]
            current = size-1
            seq_len = 1
            weight = d[current]
            final_seq[0] = best_labels[size-1]

            ## need to keep track of scores on each node
            while True:
                current = p[current]
                out_seq[seq_len] = current
                node_scores[seq_len] = d[current]
                final_seq[seq_len]   = best_labels[current]
                if current <= source: break
                seq_len += 1

        ## check if it is invalid
        ## compute resulting sequences 
        end_seq = np.array([i for i in out_seq[:seq_len+1]],dtype=np.int32)[::-1]

        fnode_scores = np.array(
            [np.array([best_seq[(i*esize)+j] for j in range(esize)],dtype='d') for i in end_seq],
            dtype='d'
        )

        ## only go to :seq_len because last symbol is a special graph token *END*
        encoded_sequence = np.array([i for i in final_seq[:seq_len]],dtype=np.int32)[::-1][:seq_len-1]
        ## unicode sequence
        unicode_seq =  np.array([emap[(end_seq[i],end_seq[i+1])] for i in range(seq_len-1)],dtype=np.object)

        return SequencePath(
            end_seq,
            encoded_sequence,
            unicode_seq,
            fnode_scores,
            weight,
            seq_len+1
        )

    finally:
        free(d)
        free(p)
        free(node_scores)
        free(out_seq)
        free(best_seq)
        free(nseq_len)
        free(best_labels)
        free(final_labels)
        free(final_seq)

## CLI STUFF

## CLI

def params():
    """main parameters for running the aligners and/or aligner experiments

    :rtype: tuple
    :returns: description of option types for symmetric aligner
    """
    from zubr.SymmetricAlignment import params
    from zubr.ExecutableModel import params as eparams
    
    aligner_group,aligner_param = params()
    e_group,e_params = eparams()
    aligner_group["GraphDecoder"]      = "Settings for the graph decoder"
    aligner_group["PolyglotDecoder"]   = "Settings for polyglot decoder (if used)"
    aligner_group["GeneralDecoder"]    = "General settings for decoders"
    aligner_group["ConcurrentDecoder"] = "Settings for running concurrent decoder"
    
    options = [
        ("--k","k",100,"int",
         "The size of shortest paths to generate  [default=100]","GraphDecoder"),
        ("--decoder_type","decoder_type","wordgraph","str",
         "The type of graph decoder to use [default='wordgraph']","GraphDecoder"),
        ("--graph","graph","","str",
         "The path to the underlying graph [default='']","GraphDecoder"),
        ("--spec_lang","spec_lang",False,"bool",
         "Specify the language when predicting [default=False]","PolyglotDecoder"),
        ("--eval_set","eval_set",'valid',"str",
            "the evaluation set [default='validation']","GeneralDecoder"),
        ("--num_jobs","num_jobs",2,"int",
            "The number of concurrent jobs to run [default=2]","ConcurrentDecoder"),
        ("--run_model","run_model",False,"bool",
            "The model to run [default=False]","ConcurrentDecoder"),
        ("--jlog","jlog",'',"str",
            "The logger path for running subjobs [default='']","ConcurrentDecoder"),
        ("--decode_data","decode_data",True,"bool",
            "Decode some data after loading [default=False]","ConcurrentDecoder"),
        ("--decode_all","decode_all",False,"bool",
            "Decode all available data [default=False]","ConcurrentDecoder"),
        ("--hoffman","hoffman",False,"bool",
            "Run the fater hoffmann k-shortest path [default=False]","ConcurrentDecoder"),
        # ("--lang_","hoffman",False,"bool",
        #     "Run the fater hoffmann k-shortest path [default=False]","ConcurrentDecoder"),            
    ]

    options += aligner_param
    options += e_params
    return (aligner_group,options)

def argparser():
    """Return an aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    
    usage = """python -m zubr graphdecoder [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

    
def main(argv):
    """The main runtime function

    :param argv: cli input or a configuration
    :type argv: list of zubr.util.config.ConfigAttrs
    :rtype: None 
    """
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])
        if config.jlog:
            logging.basicConfig(filename=config.jlog,level=logging.DEBUG)
        else: 
            logging.basicConfig(level=logging.DEBUG)
    try:
        ## run an existing model 
        if config.run_model:
            ## set directory to place where data sits
            config.dir = os.path.dirname(config.atraining)
            ## point to rank file 
            config.rfile = os.path.join(config.dir,"rank_list.txt")
            ## decoder type
            decoder_class = Decoder(config.decoder_type)
            ## load the decoder from dependencies
            decoder = decoder_class.load_backup(config)
            #decoder = decoder_class.load_large(config.run_model)
            try: 
                decoder.decode_data(config)
            except Exception,e:
                decoder.logger.error(e,exc_info=True)

        ## train model and evaluate 
        else: 
            decoder_class = Decoder(config.decoder_type)
    
            with decoder_class.from_config(config) as decoder:
                #decoder = decoder_class.from_config(config)
                decoder.logger.info('Loaded the graph decoder...')
                # ## train decoder 
                decoder.train(config)

                ## decode all data (all = training and development)
                if config.decode_all:
                    try:

                        oeval_set = config.eval_set
                        ## rank files 
                        ext = os.path.join(config.dir,"ranks.txt")
                        tr  = os.path.join(config.dir,"train_ranks.txt")
                        vr  = os.path.join(config.dir,"valid_ranks.txt")
                        ter  = os.path.join(config.dir,"valid_ranks.txt")                        
                        rr = os.path.join(config.dir,"rank_results.txt")
                        train_results = os.path.join(config.dir,"rank_results_train.txt")
                        #train_results = os.path.join(config.dir,"rank_results_test.txt")

                        ## first training data 
                        config.eval_set = 'train'
                        decoder.decode_data(config)
                        ## replace ranks
                        shutil.copy(ext,tr)
                        config.train_ranks = tr
                        try:
                            shutil.copy(rr,train_results)
                            shutil.remove(rr)
                        except: pass

                        ## second validation data
                        #config.eval_set = 'valid' if oeval_set == 'valid' else 'test'
                        config.eval_set = 'valid'
                        decoder.decode_data(config)
                        ## replace ranks
                        shutil.copy(ext,vr)
                        config.valid_ranks = vr


                        ## finally the testing data
                        if oeval_set == 'test':
                            config.eval_set = 'test'
                            decoder.decode_data(config)
                            
                            ## replace ranks
                            shutil.copy(ext,ter)

                    except Exception,e:
                        decoder.logger.error(e,exc_info=True)

                ## decode some particular set
                elif config.decode_data: 
                    try: 
                        decoder.decode_data(config)
                    except Exception,e:
                        decoder.logger.error(e,exc_info=True)

                ## backup the decoder
                decoder.logger.info('Backing up the trained model...')
                decoder.backup(config.dir)

    except Exception,e:
        traceback.print_exc(file=sys.stdout)
    finally:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
