#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Classes for querying various zubr utilities (e.g. SemAligner
in an external appication)

"""

import random
import sys
import os
import traceback
import logging
import time
import pickle
from cython cimport boundscheck,wraparound,cdivision
from zubr.Aligner cimport Model1,_Aligner as find_aligner
from zubr.util.aligner_util import build_query_data
import numpy as np
cimport numpy as np


##################################
############### Base query class

cdef class BaseQueryObj:

    """Base Class for query objects"""

     
    cpdef list query(self,qinput,int size=10):
        raise NotImplementedError()

    cdef inline unicode to_unicode(self,s):
        """converting input to unicode"""
        if isinstance(s,bytes):
            return (<bytes>s).decode('utf-8')
        return s

    property name:
        """QueryObject name or identifier"""
        def __get__(self):
            return <str>self._name
        
        def __set__(self,new_name):
            self._name == new_name

    @classmethod
    def from_mode(cls,path):
        """load a base query model

        :param path: path to model file
        :rtype: BaseQueryObj
        :returns: loaded model
        """
        with open(path) as my_model:
            return pickle.load(my_model)

    @classmethod
    def setup_query(cls,config):
        raise NotImplementedError()

    def dump_query(self,out_path):
        self._logger.info('pickling aligner model: %s' % out_path)
        with open(out_path,'wb') as my_path:
            pickle.dump(self,my_path)

    @property
    def _logger(self):
        """class logger

        :returns: logger object
        """
        level = '.'.join([__name__,type(self).__name__]) 
        return logging.getLogger(level)

    def train(self,config):
        """train the base model

        :param config: overall configuration file
        :rtype: None
        """
        raise NotImplementedError()

    def random_text(self):
        """generate a random example

        :returns: a random str
        :rtype: str
        """
        raise NotImplementedError()

cdef class M1Query(BaseQueryObj):
    """query object for IBM model1 sem aligner"""

    def __init__(self,name,aligner,ranks,rank_html,lower=True,
                 encoding='utf-8'):
        """

        :param name: name/id of the aligner
        :param aligner: model 1 base model
        :param ranks: list of items to rank
        :param rank_str: str representation of rank list
        :param rank_html: links related to rank items
        """
        if not isinstance(aligner,Model1):
            raise ValueError('only model1 supported...')
        
        self._name     = name
        self.aligner   = aligner
        self.ranks     = ranks
        self.rank_html = rank_html
        self.lower     = lower
        self.encoding  = encoding

    @classmethod
    def setup_query(cls,config):
        """set up a query object

        :param config: configuration object
        :type config: zubr.util.ConfigAttrs
        :returns: a M1Query class instance
        :rtype: M1Query 
        """
        name = config.name
        encoding = config.encoding
        if config.model_exists:
            aligner = Model1.from_model(config.model_exists)
        else:
            aligner = find_aligner(config)
            aligner.train(config=config)
            
        rl,uris = build_query_data(config,aligner.source_lex,aligner.target_lex)
        return cls(name,aligner,rl,uris,encoding)
       
    @staticmethod
    def get_query_data(cls,config):
        """get the rank and html formatted data (if it exists)

        :param config: configuration object
        :type config: zubr.util.ConfigAttrs
        :rtype: tuple
        :returns: rank list with html formatted strings
        """
        pass

    def glue(self,html,query,ref):
        """glues together output representation with string

        :param html: the hmtl output representation
        :param query: the actual query
        :param ref: the reference sentence from API
        """
        ref_str = '</tt><hr><tt> '
        query_words = set(query.split())
        
        for substring in ref.split():
            if substring.lower() in query_words:
                ref_str += "<span style='color: #D80000'> %s </span>" % substring
            else:
                ref_str += (" "+substring+" ")
        ref_str += "</tt><br><br>"
        overall = "%s %s" % (html,ref_str)
        overall.encode(self.encoding)
        return overall
    
    @boundscheck(False)
    @wraparound(False)
    cpdef list query(self,qinput,int size=10):
        """run a query from input through encoding, to a ranked list

        :param input: input string (or unicode)
        """
        cdef unicode query
        cdef dict lex = self.aligner.target_lex
        cdef int[:] top
        cdef double t,t2
        cdef int j
        cdef list reps = self.rank_html
        cdef list output = ["None"]*size
        cdef bint lower = self.lower

        if lower:
            qintput = qinput.lower()
            
        query = self.to_unicode(qinput)
        en = np.array([lex.get(i,-1) for i in query.split()],dtype='int32')
        t = time.time()
        top = self._query(en,size)
        t2 = time.time()-t

        for j in range(size):
            html1,ref = reps[top[j]]
            output[j] = self.glue(html1,query,ref)
        return output

    def random_text(self):
        """returns a random example

        :returns: a random string
        :rtype: str
        """
        rand = random.choice(self.rank_html)[-1]
        return str(rand.decode(self.encoding))
    
    @boundscheck(False)
    @wraparound(False)
    cdef np.ndarray _query(self,int[:] en,int size):
        """run a query

        :param en: english input 
        """
        cdef Model1 aligner = self.aligner
        cdef np.ndarray ranks = self.ranks
        cdef int rank_size = ranks.shape[0]
        cdef np.ndarray[ndim=1,dtype=np.int32_t] sort_list

        sort_list = np.ndarray((rank_size,),dtype='int32')
        sort_list.fill(-1)
        self.aligner._rank(en,ranks,sort_list)
        return sort_list[:size]

    def __reduce__(self):
        return M1Query,(self._name,self.aligner,self.ranks,
                        self.rank_html,self.lower,self.encoding)

cpdef Query(config):
    """factory for assinging query object type

    :param config: main configuration object
    :type config: zubr.util.ConfigAttrs
    """
    qtype = config.query_type.lower()
    if qtype == "aligner":
        ## note, this will train model if model doesnt exist
        return M1Query.setup_query(config)
     
    raise ValueError('query object not known: %s' % qtype)
    

def params():
    from zubr.Aligner import params
    aligner_group,aligner_param = params()
    aligner_group["QueryEr"] = "Settings for query object"

    options = [
        ("--name","name",'generic',"str",
         "name of query object [default='generic']","QuerEr"),
        ("--query_type","query_type","aligner","str",
         "type of query model [default=aligner]","QuerEr"),
        ("--html_rep","html_rep",True,"bool",
         "use html target representation [default=True]","QuerEr"),
        ("--data_path","data_path",'','str',
         "location of the pre-built data (if exists) [default='']","QuerEr"),
        ("--model_exists","model_exists",'','str',
         "location of the pre-build model (if exists) [default='']","QuerEr"),
    ]

    options += aligner_param
    return (aligner_group,options)

def argparser():
    """return an aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr query [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser 

def main(argv):
    from zubr.util import ConfigAttrs
    
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv)
        logging.basicConfig(level=logging.DEBUG)
        
    try:
        query_obj = Query(config)
        # if config.backup:
        #     model_out = os.path.join(config.dir,"query.model")
        #     query_obj.dump_query(model_out)
    except Exception,e:

        traceback.print_exc(file=sys.stdout)
        sys.exit()     
    
