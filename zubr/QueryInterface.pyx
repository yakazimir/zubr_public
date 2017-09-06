#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Classes for querying rank models 

"""

import os
import sys
import re
import traceback
import time
import logging 
from cython cimport boundscheck,wraparound,cdivision
import numpy as np
cimport numpy as np
from zubr.util import ConfigObj,ConfigAttrs
from zubr.util.interface_util import *
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Extractor cimport Extractor
from zubr.Alignment cimport WordModel
from zubr.Learner   cimport OnlineLearner
from zubr.Optimizer cimport RankOptimizer
from zubr.Features  cimport FeatureObj
from zubr.Dataset   cimport RankDataset,RankComparison,RankPair 

cdef class BaseQueryInterface(ZubrSerializable):
    """Base class for query objects"""

    def process_query(self,query):
        """Runs a cascade of simple regex-based preprocessing to lowercase,
        remove puncutation, and so on.

        To implement a new preprocessing or tokenization method, just
        re-implement this in a subclass. 

        :param query: the input query to the model 
        :type query: basestring 
        :returns: reformatted query 
        :rtype: unicode 
        """
        return __default_preprocess(query)

    cpdef QueryOutput query(self,query_input,int size=200):
        """Query the underlying model from some input

        :param query_input: the raw query input for the model 
        :returns: a list of model outputs 
        """
        raise NotImplementedError

    cpdef np.ndarray encode_input(self,rinput):
        """Encode a given input into the underlying model representation 
        
        :param rinput: the raw input to encode 
        :rtype: np.ndarray 
        """
        raise NotImplementedError

cdef class SequenceDecoderInterface(BaseQueryInterface):
    """A class that uses a sequence decoder, or generates arbitrary sequences"""

cdef class RankDecoderInterface(BaseQueryInterface):
    """A class that uses a rank decoder, or generates items from a ranked list"""

cdef class TranslationModelInterface(RankDecoderInterface):
    """Query interface that uses just the underlying translation model"""

    cpdef np.ndarray encode_input(self,rinput):
        """Encode a given input into the underlying model representation 
        
        :param rinput: the raw input to encode 
        :rtype: np.ndarray 
        """
        cdef WordModel trans = <WordModel>self.translator 
        cdef dict lex = trans.elex
        return np.array([lex.get(i,-1) for i in rinput.split()],dtype=np.int32)

cdef class RerankerDecoderInterface(RankDecoderInterface):
    """Query interface that uses rank decoder with reranker"""

    def __init__(self,model,extractor,rank_reps,rank_size=200):
        """Creates a RerankerDecoderInterface instance from a model and extractor 

        :param model: the underlying reranking model 
        :param extractor: the underlying features extractor 
        """
        self.model = model
        self.extractor = extractor
        self.rank_size = rank_size
        self._rank_reps = rank_reps

    cpdef np.ndarray encode_input(self,rinput):
        """Encode a given input into the underlying model representation 
        
        :param rinput: the raw input to encode 
        :rtype: np.ndarray 
        """
        cdef Extractor extractor = <Extractor>self.extractor
        cdef WordModel trans     = <WordModel>extractor.base_model
        cdef dict lex = trans.elex
        
        return np.array([lex.get(i,-1) for i in rinput.split()],dtype=np.int32)

    def process_output(self,output,surface,size,time):
        """Process the output generated from the model .

        :param output: the output generated from the model. 
        :param surface: the query surface form (if needed) 
        :param size: the size of the required output 
        """
        raise NotImplementedError
    
    cpdef QueryOutput query(self,query_input,int size=20):
        """Query the underlying model from some input

        Given the current design of the features extractor class, 
        it's a bit tricky to get the reranker to run on a single 
        example, it's more set up for extracting from complete
        Datasets. This should be changed. 

        :param query_input: the raw query input for the model 
        :returns: a list of model outputs 
        """
        cdef Extractor extractor = <Extractor>self.extractor
        cdef OnlineLearner model = <OnlineLearner>self.model
        cdef unicode uinput = self.process_query(query_input)
        cdef np.ndarray encoded = self.encode_input(uinput)
        cdef RankDataset data
        cdef RankComparison new_ranks
        cdef RankPair instance
        cdef int[:] candidates

        st = time.time()
        ## build a dataset representation of query 
        data = RankDataset.single_dataset(encoded,uinput)
        instance = data.get_item(0)

        ## build the underlying ranks 
        extractor.offline_init(data,'query')
        new_ranks = extractor.rank_init('query')

        ## extract features
        features = extractor.extract_query(instance,'query')

        ## run with discriminative model
        model.score_example(instance,0,features,new_ranks)
        candidates =  new_ranks.new_ranks[0]
        
        return self.process_output(candidates,uinput,size,time.time()-st)


cdef class HTMLRerankerInterface(RerankerDecoderInterface):
    """A reranker interface that returns html representations 

    This classes assumes that there is a file: rank_list_uri.txt, in
    the working directory that already has the output list formatted 
    as html. 
    """

    @classmethod
    def from_config(cls,config):
        """Build a html reranker interface from a configuration

        :param config: the target configuration 
        """
        cdef RankOptimizer optimizer
        
        ## rank list information
        rank_rep = read_rank_file(config,'html')
        
        ## model information 
        model_loc = config.qmodel
        optimizer = <RankOptimizer>RankOptimizer.load_large(model_loc)

        return cls(optimizer.model,optimizer.extractor,rank_rep)

    def process_output(self,output,surface,size,time):
        """Process the output generated from the model .

        :param output: the output generated from the model. 
        :param surface: the query surface form (if needed) 
        :param size: the size of the required output 
        """
        html_out = ''
        input_set = surface.lower().split()

        ## go through each item and format it 
        for item in output[:size]:
            location,reference = self._rank_reps[item]
            ref_str = '</tt><hr><tt> '
            
            ## highligh word overlap
            for word in reference.split():
                if word.lower() in input_set:
                    ref_str += "<span style='color: #D80000'> %s </span>" % word
                else:
                    ref_str += (" "+word+" ")
            ref_str += "</tt><br><br>"
            html_out += "%s %s\n\n" % (location,ref_str)

        raw_indices = [i for i in output[:size]]
        return QueryOutput(html_out,time,raw_indices)
        #return QueryOutput(html_out,time)
        

    def __reduce__(self):
        ## pickle implementation
        return HTMLRerankerInterface,(self.model,self.extractor,self._rank_reps,self.rank_size)


cdef class TextRerankerInterface(RerankerDecoderInterface):
    """A reranker interface that returns html representations 

    This classes assumes that there is a file: rank_list.txt, in
    the working directory that already has the output list formatted 
    according to the desired output 
    """

    @classmethod
    def from_config(cls,config):
        """Build a html reranker interface from a configuration

        :param config: the target configuration 
        """
        cdef RankOptimizer optimizer

        ## rank list information
        rank_rep = read_rank_file(config,'string')
        
        ## model information 
        model_loc = config.qmodel
        optimizer = <RankOptimizer>RankOptimizer.load_large(model_loc)

        return cls(optimizer.model,optimizer.extractor,rank_rep)

    def process_output(self,output,surface,size,time):
        """Process the output generated from the model .

        :param output: the output generated from the model. 
        :type query: numpy array 
        """
        reps = [self._rank_reps[i] for i in output[:size]]
        
        return QueryOutput(reps,time,output[:size].tolist())

    def __reduce__(self):
        ## pickle implementation
        return TextRerankerInterface,(self.model,self.extractor,self._rank_reps,self.rank_size)


## QUERY CLASS

cdef class QueryOutput:

    """The output of a the query interface"""

    def __init__(self,rep='',time=0.0,ids=[]):
        """Initialize a query output object
        
        """
        self.rep  = rep
        self.time = time
        self.ids  = ids


## IMPLEMENTATIONS

## regexes

comma    = re.compile(r'([a-zA-Z0-9])(\,|\;|\:)')
paren1   = re.compile(r'\(([a-zA-Z0-9\s\-\+\.]+)\)')
paren2   = re.compile(r'\[([a-zA-Z0-9\s\-\+\.]+)\]')
punc1    = re.compile(r'\s(\,|\)|\(|\?)\s')
punc2    = re.compile(r'\s(\,|\)|\(|\?|\.)$')
punc3    = re.compile(r'(\?|\!|\.|\;|\n|\\n)$')
quote1   = re.compile(r'\'([a-zA-Z\s\-]+)\'')
quote3   = re.compile(r'\"([a-zA-Z\s\-\!]+)\"')
quote2   = re.compile(r'\`|\'|\"+')
greater  = re.compile(r'&gt')
lessthan = re.compile(r'&lt')

## unicode checker function

cdef unicode to_unicode(s):
    if isinstance(s,bytes):
        return (<bytes>s).decode('utf-8')
    return s

cpdef unicode __default_preprocess(query):
    """Default preprocessing pipeline 

    :param query: the query to preprocess
    """
    cdef unicode text = to_unicode(query)
    text = text.lower()
    text = re.sub(comma,r'\1 ',text)
    text = re.sub(r'\?\s*$','',text)
    text = re.sub(paren1,r' \1 ',text)   
    text = re.sub(r' \( | \) ',' ',text)
    text = re.sub(punc1,r' ',text)
    text = re.sub(punc2,r' ',text)
    text = re.sub(r'\`+','',text).strip()    
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\.$','',text).strip()
    return text

## TRANSLATION IMPLEMENTATIONS

INTERFACES = {
    "rank_html"  : HTMLRerankerInterface,
    "rank_text"  : TextRerankerInterface,
}

## interfance factory

def QueryInterface(config):
    """Factory method for finding a query interface class

    :param config: the main query configuration 
    """
    qclass = INTERFACES.get(config.interface,None)
    if not qclass:
        raise ValueError('Uknown type of query interface: %s' % config.interface)
    return qclass


def params():
    """Main parameters for building query interfaces 

    :returns: description of configuration item and configuration options 
    """
    groups = {"QueryInterface" : "Settings for query interfaces"}
    
    options = [
        ("--interface","interface","rank_html","str",
         "The type of query interface to use [default='']","QueryInterface"),
        ("--qmodel","qmodel",'',"str",
         "The location of the query model [default='']","QueryInterface"),
        ("--rfile","rfile",'',"str",
         "The location of the rank file [default='']","QueryInterface"),
        ("--same_dir","same_dir",False,"bool",
         "Build interface in same directory as model [default='']","QueryInterface"),
    ]

    return (groups,options)
        

def argparser():
    """Returns the query interface configuration argument parser 

    :rtype: zubr.util.config.ConfigObj 
    :returns: default argument parser 
    """
    from zubr import _heading
    from _version import __version__ as v

    usage = """python -m zubr queryinterface [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser
    

def main(argv):
    """"The main entry point for using these sets of classes 

    :param argv: the cli input or a configuration object 
    """
    interface_logger = logging.getLogger('zubr.QueryInterface.main')
    
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])
        logging.basicConfig(level=logging.DEBUG)
        ## set the directory path 
        if not config.dir:
            config.dir = os.path.dirname(config.qmodel)

    qclass = QueryInterface(config)

    try:
        interface_logger.info('Building interface and loading model, might take a moment...')
        query_instance = qclass.from_config(config)
        
    except Exception,e:
        traceback.print_exc(file=sys.stdout)
    finally:

        ## backup the query item 
        try:
            if config.dir:
                out_path = os.path.join(config.dir,"query")
                #query_instance.dump_large(out_path)
                ## somehow the dump_large (above) is corrupting the files 
                query_instance.dump(out_path)
                ## put in pointer to use with query server
        except:
            pass
                

if __name__ == "__main__":
    main(sys.argv[1:])
