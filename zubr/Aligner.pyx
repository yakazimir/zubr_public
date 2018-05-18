#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementation of IBM Models 1 and 2

"""

import os
import sys
import time
import logging
import traceback
from zubr.util.aligner_util import load_aligner_data,get_rdata,get_tree_data,get_test_data
from zubr.util.aligner_util import get_tree_data
from zubr.Alg cimport binary_insert_sort 
from cython cimport boundscheck,wraparound,cdivision
cimport numpy as np
import numpy as np
from collections import Counter
from cpython cimport PyList_GET_SIZE,PyList_Append
import pickle
import gzip
from zubr.ZubrClass cimport ZubrSerializable

empty = np.ndarray((2,0))
empty2 = np.ndarray((1,),dtype='d')

class AlignerError(Exception):
    pass 

cdef class AlignerBase:

    """Base aligner class"""

    def __init__(self,source_lex={},target_lex={},max_len=30,
                 stops=False,constraints=False,encoding='utf-8',
                 table=empty,wc=None,sc=None,alambda=0.0,minprob=0.0,
                 ignoreoov=False):
        """

        :param stops: remove stop words when aligning
        :type stops: bool
        :param constraints: training aligner using constraints
        :type contraints: bool
        """
        self.stops = stops
        self.max_len = max_len
        self.source_lex = source_lex
        self.target_lex = target_lex
        self.encoding = encoding
        self.table = table
        self.alambda = alambda
        self.minprob = minprob
        self.ignoreoov = ignoreoov

        ## word frequency
        if not wc:
            self._wc = empty2
            self._wc.fill(0.0)
        else:
            self._wc = wc

        ## symbol frequency
        if not sc:
            self._sc = empty2
        else:
            self._sc = sc
            
    def train(self,config):
        """main python method for training model

        :param config: aligner configuration
        :type config: zubr.util.config.ConfigAttrs
        """
        raise NotImplementedError()

    def decode(self,f,e):
        """python method for decoding input

        :param source: source aligner input,e
        :param target: target aligner input,f
        :returns: aligner string with probability
        :rtype: tuple 
        """
        raise NotImplementedError()

    @classmethod
    def from_model(cls,path):
        """loading aligner from file

        :param path: path where the aligner model is
        :type path: str
        :returns: loaded aligner
        :rtype: AlignerBase 
        """
        with gzip.open(path,'rb') as my_model:
            return pickle.load(my_model)

    @classmethod
    def load_data(cls,path='',config=None):
        """load an aligner with either a configuration
        object or a path to data

        :param path: path to alignment data
        :type path: str
        :param config: configuration object
        :type config: zubr.util.config.ConfigAttrs
        :rtype: None

        >>> from zubr.Aligner import BaseAligner 
        >>> a = BaseAligner(path='examples/hansards')
        """
        if config == None:
            parser = argparser() 
            config = parser.get_default_values()
        
        if path: config.align_training = path
        _,_,sdict,tdict,table = load_aligner_data(config)
        return cls(source_lex=sdict,target_lex=tdict,
                   max_len=config.amax,encoding=config.encode,
                   stops=config.stops,table=table)

    def dump_aligner(self,out_path):
        """pickle the given aligner model

        :param out_path: path to put the model
        :param out_path: str
        :rtype: None        
        """
        self._logger.info('pickling aligner model: %s' % out_path)
        with gzip.open(out_path,'wb') as my_path:
            pickle.dump(self,my_path)
            
    @property
    def _logger(self):
        level = '.'.join([__name__,type(self).__name__]) 
        return logging.getLogger(level)
    
    def word_map(self):
        """reverse the source and target lexicons

        :returns: word side of source/target lexicons
        :rtype: dict(dict,dict)
        """
        return {"s":{k[1]:k[0] for k in self.source_lex.items()},
                "t":{k[1]:k[0] for k in self.target_lex.items()}}
    
    def print_table(self,file=sys.stdout,minp=0.0001):
        """print the co-occurence probabilities

        :param file: where to the table to
        :type file: file or sys.stdout
        :rtype: None
        """
        words = self.word_map()
        source_len = self.table.shape[0]
        target_len = self.table[0].shape[0]
        self._logger.info('printing probability table')
        
        for sid in range(source_len):
            print >>file, "f: %s" % words["s"][sid].encode('utf-8')
            wl = {words["t"][k]:i for k,i in enumerate(self.table[sid])}
            for w,prob in Counter(wl).most_common():
                if prob < minp or prob <= 0.0: continue
                print >>file,"\t\t%s\t%f" % (w.encode('utf-8'),prob)

    def find_viterbi(self,f,e):
        """find the vitebri alignment for two inputs

        :param f: foreign side
        :type f: str
        :param e: english side
        :type e: str
        :raises: AlignerError
        """
        if not isinstance(f,basestring) or\
          not isinstance(e,basestring):
          raise AlignerError('wrong viterbi input')

        if not isinstance(e,unicode):
            e = unicode(e,'utf-8')

        if not isinstance(f,unicode):
            f = unicode(f,'utf-8')
            
        ei = np.array([self.target_lex.get(w,-1) for w in e.split()],dtype=np.int32)
        fi = np.array([0]+[self.source_lex.get(w,-1) for w in f.split()],dtype=np.int32)
        return self._decode(fi,ei).alignments()[0]

    property flex:
        def __get__(self):
            return <dict>(self.source_lex)

    property elex:
        def __get__(self):
            return <dict>(self.target_lex)
                
    property source_len:
        """return the length of the source vocabulary"""
        def __get__(self):
            return <int>len(self.source_lex)

    property target_len:
        """return the length of the target vocabulary"""
        def __get__(self):
            return <int>len(self.target_lex)

    property word_counts:
        """word counts gathered from training corpus"""
        def __get__(self):
            return self._wc

    property symbol_counts:
        def __get__(self):
            return self._sc
        
    cpdef decode_dataset(self,sourced,targetd,out=None,k=1):
        """decode a dataset (train or test) 

        :param sourced: source training set
        :type sourced: np.array
        :param targetd: target training set
        :type targetd: np.array
        :param out: place to print alignments to
        :type out: str or sys.stdout
        :param k: number of aligments per input (default=1 best)
        :type k: int
        :rtype: None
        """
        cdef unsigned int slen = sourced.shape[0]
        cdef unsigned int tlen = targetd.shape[0]
        cdef int i
        cdef Decoding alignment
        out = sys.stdout if out == None else open(out,"w")
        
        for i in range(slen):
            for alignment in self._decode(sourced[i],targetd[i]):
                print >>out,alignment

        # close the file
        if out != sys.stdout:
            out.close()

    @boundscheck(False)
    @wraparound(False)
    cpdef tuple rank_dataset(self,object config):
        """rank example input/output pairs for a given test datset

        :param config: aligner configuration object
        :type config: zubr.util.config.ConfigAttrs
        :returns: array of gold rep paired with ranked list of reps.
        :rtype: tuple
        """
        cdef np.ndarray en,rl,freq,order
        cdef np.ndarray[dtype=np.int32_t,ndim=2] ranks
        cdef np.ndarray[dtype=np.int32_t,ndim=1] gid
        cdef int e,i,j,r
        cdef int rsize = config.ranksize

        rl,inp,order,freq,enorig = <tuple>get_rdata(config,self.flex,self.elex)
        en,gid = inp
        e = en.shape[0]
        r = rl.shape[0]
        ranks = np.ndarray((e,r),dtype='int32')
        ranks.fill(-1) 
        for i in range(e):
            self._rank(en[i],rl,ranks[i])

        return (rl,ranks,gid,order,enorig)

    ### rank c function
    #########################

    @cdivision(True)
    @boundscheck(False)
    cdef _rank(self,int[:] en,np.ndarray rl,int[:] sort):
        """decode a list of e sequences using a list of f sequences rl, then sort

        -- sorting is done using a binary insertion search
        
        :param en: list of english inputs to decode
        :type en: np.ndarray
        :param rl: list of foreign inputs to decode
        :type rl: np.ndarray
        """
        cdef int r = rl.shape[0]
        cdef int i,end,start,mid
        cdef double prob,sprob
        cdef Alignment alignment
        #cdef unsigned int curr = 0
        cdef double[:] problist = np.ndarray((r,),dtype='d')
        
        problist[:] = 0.0
        ## get a list of word counts
                
        for i in range(r):
            alignment = self._decode(rl[i],en)
            prob = alignment.prob
            if isnan(prob):
                raise AlignerError(
                    'encountered NaN score, is maxlen set correctly?')
            problist[i] = prob
            binary_insert_sort(i,prob,problist,sort)
            
    ### decode c function
    #########################

    cdef Alignment _decode(self,int[:] s,int[:] t):
        """main decode function

        :param s: source input
        :param t: target intput
        :param lambd: smoothing constant (defaults to 0.0)
        :returns: alignment object
        :rtype: Alignment
        """
        raise NotImplementedError

    ## estimate corpus word frequencies 
    #########################
    
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef void _compute_freq(self,np.ndarray e):
        """for a corpus, estimate word frequences on the english side"""
        cdef int i,k,ssize,wid
        cdef int setsize = e.shape[0]
        cdef double[:] wc = self._wc
        cdef int[:] instance
        cdef double total = 0.0
        cdef int vocabsize = self.target_len

        if vocabsize > 1:
        
            for i in range(setsize):
                instance = e[i]
                ssize = instance.shape[0]
                for k in range(ssize):
                    total += 1.0
                    wid = instance[k]
                    wc[wid] += 1.0

            for i in range(vocabsize):
                if wc[i] == 0.0: continue 
                wc[i] = (wc[i]/total)

    ## estimate corpus symbol frequencies 
    #########################
                
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef void _compute_sfreq(self,np.ndarray f):
        """for a corpus, estimate word frequences on the english side"""
        cdef int i,k,ssize,wid
        cdef int setsize = f.shape[0]
        cdef double[:] sc = self._sc
        cdef int[:] instance
        cdef double total = 0.0
        cdef int vocabsize = self.source_len

        if vocabsize > 1:
        
            for i in range(setsize):
                instance = f[i]
                ssize = instance.shape[0]
                for k in range(1,ssize):
                    total += 1.0
                    wid = instance[k]
                    sc[wid] += 1.0

            for i in range(vocabsize):
                if sc[i] == 0.0: continue 
                sc[i] = (sc[i]/total)
    
                            
    ## implementation of each train strategy
    #########################

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef void train_model1(self,np.ndarray source,np.ndarray target,int maxiter,
                      double smooth=0.0):
        """method for training IBM model1

        :param source: source side training data
        :param target: target side training data
        :param maxiter: maximum iterations
        :param smooth: smoothing constant 
        """
        cdef int size = source.shape[0]
        cdef int slen,tlen,k,j,e,f
        cdef unsigned int iterations = 0 
        cdef int[:] sourcet
        cdef int[:] targett
        cdef double logprob = 1e-1000
        cdef double newlogprob,z,count,score,news
        cdef double total,paircount
        cdef double[:,:] table = self.table
        cdef double[:,:] counter
        cdef double[:] globalcounts
        cdef int slex = self.source_len
        cdef int tlex = self.target_len
        
        counter = np.full((slex,tlex),smooth)
        globalcounts = np.full((slex,),smooth*float(tlex)) 

        while True:
            iterations += 1
            if iterations > maxiter: break
            if iterations > 1: logprob = newlogprob
            newlogprob = 1e-1000

            # reset counters
            counter[:,:] = smooth
            globalcounts[:] = (smooth*tlex)
            
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
                        z += table[sourcet[j],e]
                    for j in range(slen):
                        f = sourcet[j]
                        count = table[f,e]/z
                        counter[f,e] += count
                        globalcounts[f] += count
                        
                    newlogprob += log(z)

            #m-step
            for i in range(slex):
                total = globalcounts[i]
                ## ignore unobserved items?
                if total == (smooth*float(tlex)):
                    continue 

                for k in range(tlex):
                    paircount = counter[i,k]
                    if paircount == 0.0:
                        table[i,k] = 0.0
                    else:
                        table[i,k] = (paircount/total)
                
            ## report on likelihood
            if iterations > 2 and logprob > newlogprob:
                break

            self._logger.info("Iteration: %d,\t log likelihood: %f" \
                               % (iterations,newlogprob))

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef void train_model2(self,np.ndarray source, np.ndarray target,
                           int maxiter, int maxl, double smooth1=0.0,
                           double smooth2=0.0):
        """method for training IBM model2, similar to model1 with the
        exception of including an extra distortion model

        :param source: source training input
        :param target: target training input
        :param maxiter: maximum number of iterations
        :param smooth1: smoothing constant for co-occurrence
        :param smooth2: smoothing constant for distortion
        """
        cdef int comparisons,size = source.shape[0]
        cdef int slen,tlen,k,j,e,f,i,K,I
        cdef unsigned int iterations = 0
        cdef int[:] sourcet
        cdef int[:] targett
        cdef double logprob = 1e-1000
        cdef double newlogprob,count,score,news
        cdef double total,paircount,z
        cdef double[:,:] table = self.table
        cdef double[:,:,:,:] distortion = self.distortion
        cdef double[:,:,:,:] dcounter
        cdef double[:,:,:] gcounts2
        cdef double[:,:] ccounter
        cdef double[:] gcounts1
        cdef int slex = self.source_len
        cdef int tlex = self.target_len

        ccounter = np.full((slex,tlex),smooth1)
        gcounts1 = np.full((slex,),smooth1*float(tlex))
        dcounter = np.full((maxl,maxl,maxl,maxl),0.0)
        gcounts2 = np.full((maxl,maxl,maxl),0.0)

        while True:
            
            iterations += 1
            if iterations > maxiter: break
            if iterations > 1: logprob = newlogprob
            newlogprob = 1e-1000

            ## reset counters
            ccounter[:,:] = smooth1
            gcounts1[:] = (smooth1*tlex)
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
                        #z += (table[sourcet[j],e])*(distortion[slen-1,tlen-1,j,k])
                        z += (table[sourcet[j],e])*(distortion[slen-1,tlen-1,k,j])

                    for j in range(slen):
                        f = sourcet[j]
                        #count = (distortion[slen-1,tlen-1,j,k]*table[f,e])/z
                        count = (distortion[slen-1,tlen-1,k,j]*table[f,e])/z
                        ccounter[f,e] += count
                        gcounts1[f] += count
                        #dcounter[slen-1,tlen-1,j,k] += count
                        #gcounts2[slen-1,tlen-1,j] += count
                        dcounter[slen-1,tlen-1,k,j] += count
                        gcounts2[slen-1,tlen-1,k] += count
                        
                    
                    newlogprob += log(z)
                    
            # m-step

            ## co-occurence updates
            for i in range(slex):
                total = gcounts1[i]

                ## ignore unobserved items
                if total == (smooth1*float(tlex)):
                    continue
                
                for k in range(tlex):
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
                        if total == 0.0:
                            continue 

                        total += (I*smooth2)
                        for j in range(I+1):
                            paircount = dcounter[K,I,k,j]
                            paircount += smooth2
                            if paircount == 0.0:
                                distortion[K,I,k,j] = 0.0
                            else:
                                distortion[K,I,k,j] = (paircount/total)
            ##
            if iterations > 2 and logprob > newlogprob: break
            self._logger.info("Iteration: %d,\t log likelihood: %f" \
                              % (iterations,newlogprob))

cdef class Model1(AlignerBase):

    def train(self,path='',config=None):
        """main python method for training (only) model1

        -- with build all the associated data using the path,
        as well as initialized model, ect..

        :param path: path to training data
        :type path: str
        :param config: aligner configuration object
        :type config: zubr.util.config.ConfigAttrs
        :type: None 
        """
        if not config:
            parser = argparser() 
            config = parser.get_default_values()

        if path:
            config.atraining = path
        self._logger.info('loading aligner data..') 
        s,t,sd,td,table,_ = load_aligner_data(config)
        self.source_lex = sd
        self.target_lex = td
        self.table = table
        self.max_len = config.amax
        self.encoding = config.encoding
        self.minprob = config.minprob
        if not self.encoding:
            self._logger.warning('no encoding set, using utf-8')
            self.encoding = 'utf-8'        

        ## will smooth when decoding?
        if config and (config.alambda or config.tlambda):
            self.alambda = config.alambda
            self._logger.info('smoothing with lambda=%f' % self.alambda)
            self._wc = np.ndarray(self.target_len,dtype='d')
            self._wc.fill(0.0) 
            self._logger.info('estimating word frequencies...')
            self._compute_freq(t)
            
        self._logger.info('training model1..')
        ttime = time.time()
        self.train_model1(s,t,config.aiters)
        self._logger.info('trained in %f seconds' % (time.time()-ttime))

        ## align training 
        if config and config.aligntraining:
            self._logger.info('aligning training data...')
            if not config.dir: dout = None
            else: dout = os.path.join(config.dir,"alignment/train_decode.txt") 
            self.decode_dataset(s,t,out=dout)

        ## align testing
        if config and config.aligntesting:
            self._logger.info('aligning testing data...')
            ft,et,_ = get_test_data(config,self.source_lex,self.target_lex)
            if not config.dir: dout = None
            else: dout = os.path.join(config.dir,"alignment/test_decode.txt")
            self.decode_dataset(ft,et,out=dout)
            
        ## print items related to model
        if config and config.backup:
            #self._logger.info('printing alignment table...')
            if not config.dir:
                sout = sys.stdout
                self._logger.info('printing alignment table...')
            else:
                #sout = os.path.join(config.dir,"alignment/train_table.txt")
                sout = os.path.join(config.align_dir,"train_table.txt")
                self._logger.info('printing alignment table to file: %s' % sout)
            self.print_table(file=open(sout,'w'))

    def decode(self,f,e):
        """python method for decoding input

        :param source: english input e
        :param target: foreign input f
        :returns: aligner string with probability
        :rtype: tuple 
        """
        return self._decode(f,e) 

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef Alignment _decode(self,int[:] s,int[:] t):
        """c method for decoding input/output for Model1. Will do jelinek-mercer
        smoothing if lamb parameter is provided and a word-frequency array

        :param s: source input
        :param t: target input 
        :returns: alignment object
        :rtype: Alignment
        """
        cdef double[:,:] table = self.table
        cdef unsigned int slen = s.shape[0]
        cdef unsigned int tlen = t.shape[0]
        cdef Alignment record = Alignment.make_empty(slen,tlen)
        cdef double[:,:] problist = record.problist
        cdef int[:] ml = record.ml
        cdef int k,i
        cdef double z,overallprob = 0.0
        cdef double score
        cdef double lamb = self.alambda
        cdef double[:] wf = self._wc
        cdef bint elam = (lamb > 0.0)
        cdef bint smooth = (elam) and (wf.shape[0] > 1)
        cdef bint oov = self.ignoreoov
        #cdef double div = float(slen+1)**float(tlen)
        cdef double div = float(slen)**float(tlen)

        for k in range(tlen):

            ## skip unknown words on e side 
            if t[k] == -1:
                continue
            
            z = 0.0
            for i in range(slen):
                ## skip over unknown on f side
                if s[i] == -1:
                    continue 
                
                score = table[s[i],t[k]]
                z += score
                problist[k][i] = score
                if score >= problist[k][ml[k]]:
                    ml[k] = i

            ## smooth : mercer-jelinek smoothing
            if smooth:
                z = ((1-lamb)*z)+(lamb*wf[k])

            ## skip over z if zero?
            if z == 0.0 and oov:
                continue

            if overallprob == 0.0:
                overallprob = z
            else:
                overallprob *= z

        #normalize for length
        overallprob = overallprob/div
        record.prob = overallprob
        return record
    
    def __reduce__(self):
        # pickle implementation 
        return Model1,(self.source_lex,self.target_lex,self.max_len,
                       self.stops,False,self.encoding,self.table,self._wc,
                       self._sc,self.alambda,self.minprob,self.ignoreoov)
    
cdef class Model2(Model1):
    """IBM Model2 aligner"""

    def __init__(self,source_lex={},target_lex={},max_len=30,
                 stops=False,constraints=False,encoding='utf-8',
                 table=empty,distortion=empty,wc=None,sc=None,alambda=0.0,minprob=0.0,
                 ignoreoov=False):
        
        self.distortion = distortion 
        Model1.__init__(self,source_lex=source_lex,target_lex=target_lex,
                        max_len=max_len,stops=stops,constraints=constraints,
                        encoding=encoding,table=table,wc=wc,sc=sc,alambda=0.0,ignoreoov=False)

    cpdef train(self,str path='',config=None):
        """main python method for training Model2,
        (standardly) initializes by running Model1 

        :param path: path to training data
        :type path: str
        :param config: aligner configuration object
        :type config: zubr.util.config.ConfigAttrs
        :rtype: None 
        """
        if config == None:
            parser = argparser() 
            config = parser.get_default_values()
    
        if path:
            config.atraining = path
            
        config.modeltype = "IBM2"
        self._logger.info('loading aligner data...')
        s,t,sd,td,table,dis = load_aligner_data(config)
        initialize_dist(dis,config.amax)
        self.source_lex = sd
        self.target_lex = td
        self.table = table
        self.distortion = dis
        self.encoding = config.encoding
        self.max_len = config.amax
        self.minprob = config.minprob
        if not self.encoding:
            self._logger.warning('no encoding set, using default (utf-8)')
            self.encoding = 'utf-8'        

        if config and (config.alambda or config.tlambda):
            self.alambda = config.alambda
            self._logger.info('lambda decoding parameter set to: %f' % self.alambda)
            self._wc = np.ndarray(self.target_len,dtype='d')
            self._wc.fill(0.0) 
            self._logger.info('estimating word frequencies...')
            self._compute_freq(t)
            
        # training 
        self._logger.info('initializing using Model1...')
        ttime = time.time()
        self.train_model1(s,t,config.aiters)
        self._logger.info('trained in %f seconds' % (time.time()-ttime))
        self._logger.info('training full model..')
        ttime2 = time.time()
        self.train_model2(s,t,config.aiters2,config.amax)
        self._logger.info('trained in %f seconds' % (time.time()-ttime2))

        if config and config.aligntraining:
            ## change it to include cases without output dir 
            self._logger.info('aligning training data...')
            if not config.dir: dout = None
            else: dout = os.path.join(config.dir,"alignment/train_decode.txt")
            self.decode_dataset(s,t,out=dout)
        if config and config.backup:
            #self._logger.info('printing alignment table...')
            if not config.dir:
                tout = sys.stdout
                self._logger.info('printing alignment table...')
            else:
                #sout = os.path.join(config.dir,"alignment/train_table.txt")
                sout = os.path.join(config.align_dir,"train_table.txt")
                self._logger.info('printing alignment table to file: %s' % sout)
            self.print_table(file=open(sout,'w'))
            
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef Alignment _decode(self,int[:] s,int[:] t):
        """c method for decoding input/output for Model1

        :param s: source input
        :param t: target input
        :returns: alignment object
        :rtype: Alignment
        """
        cdef double[:,:] table = self.table
        cdef double[:,:,:,:] distortion = self.distortion
        cdef unsigned int slen = s.shape[0]
        cdef unsigned int tlen = t.shape[0]
        cdef Alignment record = Alignment.make_empty(slen,tlen)
        cdef double[:,:] problist = record.problist
        cdef int[:] ml = record.ml
        cdef int k,j
        cdef double z,overallprob = 0.0
        cdef double score
        cdef double lamb = self.alambda
        cdef double[:] wf = self._wc
        cdef bint elam = (lamb > 0.0)
        cdef bint smooth = (elam) and (wf.shape[0] > 1)
        cdef bint unknown

        for k in range(tlen):
            ## skip unknown words
            if t[k] == -1:
                continue 
            
            z = 0.0
            for j in range(slen):
                
                if s[j] == -1:
                    continue

                #score = (table[s[j],t[k]])*(distortion[slen-1,tlen-1,j,k])
                score = (table[s[j],t[k]])*(distortion[slen-1,tlen-1,k,j])
                z += score
                problist[k][j] = score
                if score >= problist[k][ml[k]]:
                    ml[k] = j

            ## mercer-jelinek smoothing
            if smooth:
                z = ((1-lamb)*z)+(lamb*wf[k])
                    
            if overallprob == 0.0:
                overallprob = z
            else: 
                overallprob *= z

        record.prob = overallprob
        return record
                    
    def __reduce__(self):
        # pickle implementation
        return Model2,(self.source_lex,self.target_lex,self.max_len,
                       self.stops,False,self.encoding,self.table,
                       self.distortion,self._wc,self._sc,self.alambda)


##### IBM tree model


cdef class TreeModel(Model1):
    """A variant of IBM Model 2 which takes into account
    the relative tree (as opposed to word) position of words
    being aligned"""

    def __init__(self,source_lex={},target_lex={},max_len=30,
                 encoding='utf-8',table=empty,tdistortion=empty,
                 wc=None,sc=None,alambda=0.0,ignoreoov=False):
        
        self.tdistortion = tdistortion

    cpdef train(self,str path='',config=None):
        """main python method for training tree model, which
        is first initialized by running Model1

        :param path: path to training data
        :type path: str
        :param config: aligner configuration object 
        :type config: zubr.util.config.ConfigAttrs
        :rtype: None 
        """
        if config == None:
            parser = argparser()
            config = parser.get_default_values()

        if path:
            config.atraining = path

        ##
        config.modeltype = "treemodel"
        self._logger.info('loading aligner data...')
        s,t,sd,td,table,dis = load_aligner_data(config)
        self.source_lex = sd
        self.target_lex = td
        self.table = table
        self.encoding = config.encoding
        self.max_len = config.amax
        self.minprob = config.minprob
        if not self.encoding:
            self._logger.warning('no encoding set, using default (utf-8)')
            self.encoding = 'utf-8'

        if config and (config.alambda or config.tlambda):
            self.alambda = config.alambda
            self._logger.info('lambda decoding parameter set to: %f' % self.alambda)
            self._wc = np.ndarray(self.target_len,dtype='d')
            self._wc.fill(0.0) 
            self._logger.info('estimating word frequencies...')
            self._compute_freq(t)
            self._logger.info('estimating symbol frequences...')
        
        ## tree position data
        treepos = get_tree_data(config,tset="train")
        self.tdistortion = _tree_dist(config.maxtree,config.emax)
        self._logger.info('max tree length: %d' % config.maxtree)
        ## train model1
        self._logger.info('initializing using Model1...')
        ttime = time.time()
        self.train_model1(s,t,config.aiters)
        self._logger.info('trained in %f seconds' % (time.time()-ttime))
        self._logger.info('training full model...')
        ttime2 = time.time()
        # train tree model 
        self.train_tree_model(s,t,treepos,config.aiters3,config.maxtree,
                              config.emax,trainend=config.trainstop)
        self._logger.info('trained in %f seconds' % (time.time()-ttime2))

        if config and config.aligntraining:
            self._logger.info('aligning training data...')
            if not config.dir: dout = None
            else: dout = os.path.join(config.dir,"alignment/train_decode.txt")
            self.decode_tree_dataset(s,t,treepos,out=dout)

        if config and config.aligntesting:
            self._logger.info('aligning testing data...')
            ft,et,tp = get_test_data(config,self.source_lex,self.target_lex) 
            if not config.dir: dout = None
            else: dout = os.path.join(config.dir,"alignment/test_decode.txt")
            self.decode_tree_dataset(ft,et,tp,out=dout)

        if config and config.backup:
            if not config.dir:
                sout = sys.stdout
                self._logger.info('printing alignment table...')
            else:
                #sout = os.path.join(config.dir,"alignment/train_table.txt")
                sout = os.path.join(config.align_dir,"train_table.txt")
                self._logger.info('printing alignment table to file: %s' % sout)

            self.print_table(file=open(sout,'w'))


    cpdef decode_tree_dataset(self,sourced,targetd,treepos,out=None,k=1):
        """align a dataset using the tree aligne moder

        :param sourced: the f side of the translation
        :param targetd: the e side of the translation
        :param treepos: list of tree positions associated with data
        :param out: place where to print alignments
        """
        cdef unsigned int slen = sourced.shape[0]
        cdef unsigned int tlen = targetd.shape[0]
        cdef unsigned int ntrees = treepos.shape[0]
        cdef int i,treelen
        cdef Decoding alignment
        cdef int[:] positions
        out = sys.stdout if out == None else open(out,"w")

        for i in range(ntrees):
            positions = treepos[i][:-1]
            treelen = treepos[i][-1]
            for alignment in self._tree_decode(sourced[i],targetd[i],
                                               positions,treelen):
                print >>out,alignment

        ## close this file
        if out != sys.stdout:
            out.close()
    

    cpdef tuple rank_dataset(self,object config):
        """rank input/output pairs give a list of possible rank candidates

        
        :param config: main aligner and experiment configuration
        :rtype: tuple 
        """
        cdef np.ndarray en,rl,freq,order,rtrees
        cdef np.ndarray[dtype=np.int32_t,ndim=2] ranks
        cdef np.ndarray[dtype=np.int32_t,ndim=1] gid
        cdef e,i,j,r
        cdef int rsize = config.ranksize

        rl,inp,order,freq,enorig = <tuple>get_rdata(config,self.flex,self.elex)
        rtrees = get_tree_data(config,tset="rank")
        en,gid = inp
        e = en.shape[0]
        r = rl.shape[0]
        ranks = np.ndarray((e,r),dtype='int32')
        ranks.fill(-1)
        for i in range(e):
            self._rank_tree(en[i],rl,rtrees,ranks[i])

        return (rl,ranks,gid,order,enorig)

    cdef _rank_tree(self,int[:] en,np.ndarray rl,np.ndarray treepos,int[:] sort):
        """rank a list of f outputs given an english input en

        :param en: english (or e) input
        :param rl: a list of f outputs to alignment with e and rank
        :param treepos: abstract tree positions for rank list   
        """
        cdef int r = rl.shape[0]
        cdef int i,end,start,mid,tlen
        cdef double prob,sprob
        cdef Alignment alignment
        cdef double[:] problist = np.ndarray((r,),dtype='d')
        
        problist[:] = 0.0

        for i in range(r):
            alignment = self._tree_decode(rl[i],en,treepos[i][:-1],treepos[i][-1])
            prob = alignment.prob
            if isnan(prob):
                raise AlignerError('encountered NaN score, is maxlen set correctly?')
            problist[i] = prob
            binary_insert_sort(i,prob,problist,sort)
            
    #@boundscheck(False)
    #@wraparound(False)
    @cdivision(True)
    cdef int train_tree_model(self,np.ndarray f,np.ndarray e,
                               np.ndarray tree_pos,int maxiter,int tmax,
                               int emax,int trainend=100000000,
                               double smooth1=0.0,double smooth2=0.0) except -1:
        cdef int i,j,k,size = f.shape[0]
        cdef int[:] et,ft
        cdef double z,newlogprob,logprob = 1e-1000
        cdef double[:,:,:] treedist = self.tdistortion
        cdef double[:,:] table = self.table
        cdef double[:,:] ccounter
        cdef double[:,:,:] dcounter
        cdef double[:] gcounts1
        cdef double[:,:] gcounts2
        cdef int[:] ftreepos
        cdef unsigned int iterations = 0 
        cdef unsigned int flen = self.source_len
        cdef unsigned int elen = self.target_len
        cdef unsigned int ftlen,etlen
        cdef unsigned int ftreelen
        cdef int ex,fx,K,I
        cdef double total,paircount
        cdef int poslen = tree_pos.shape[0]

        ccounter = np.full((flen,elen),smooth1)
        dcounter = np.full((tmax,emax,tmax),0.0)
        gcounts1 = np.full((flen,),smooth1*float(elen))
        gcounts2 = np.full((tmax,emax),0.0)
        
        while True:

            iterations += 1
            if iterations > maxiter:
                break

            if iterations > 1:
                logprob = newlogprob

            newlogprob = 1e-1000

            ##reset counters
            ccounter[:,:] = smooth1
            gcounts1[:] = (smooth1*flen)
            dcounter[:,:,:] = 0.0
            gcounts2[:,:] = 0.0
            
            ## e-step
            #for i in range(size):
            for i in range(poslen):

                if i >= poslen:
                    continue 
                
                ft = f[i]                
                et = e[i]
                ftreepos = tree_pos[i][:-1]
                ftreelen = tree_pos[i][-1]
                ftlen = ft.shape[0]
                etlen = et.shape[0]
                for k in range(etlen):
                    ex = et[k]
                    z = 0.0

                    for j in range(ftlen):
                        try: 
                            z += (table[ft[j],ex])*(treedist[ftreelen,k,ftreepos[j]])
                        except Exception as e:
                            #print i 
                            raise(e)

                    for j in range(ftlen):
                        fx = ft[j]
                        count = (table[ft[j],ex]*(treedist[ftreelen,k,ftreepos[j]]))/z
                        ccounter[fx,ex] += count
                        gcounts1[fx] += count
                        dcounter[ftreelen,k,ftreepos[j]] += count
                        gcounts2[ftreelen,k] += count
                        
                    newlogprob += log(z)

            ## m-step

            ## lex co-occurence updates
            
            for i in range(flen):
                total = gcounts1[i]

                if total == (smooth1*float(elen)):
                    continue

                for k in range(elen):
                    paircount = ccounter[i,k]
                    if paircount == 0.0:
                        table[i,k] = 0.0
                    else:
                        table[i,k] = (paircount/total)

            ## distortion updates (without smoothing) 

            for K in range(tmax):
                for I in range(emax):
                    total = gcounts2[K,I]
                    if total == 0.0:
                        continue

                    for j in range(tmax):
                        paircount = dcounter[K,I,j]
                        if paircount == 0.0:
                            treedist[K,I,j] = 0.0
                        else:
                            treedist[K,I,j] = (paircount/total)
                            

            if iterations > 2 and logprob > newlogprob:
                break

            self._logger.info("Iteration %d,\t log likelihood: %f" %\
                              (iterations,newlogprob))

    @cdivision(True)
    @boundscheck(False)
    cdef Alignment _tree_decode(self,int[:] s,int[:] t,int[:] treepos,int treelen):
        """c method for decoding intput/output for the TreeModel

        :param s: f input
        :param t: e input
        :rtype: Alignment 
        """
        cdef double[:,:] table = self.table
        cdef double[:,:,:] treedist = self.tdistortion
        cdef unsigned int flen = s.shape[0]
        cdef unsigned int elen = t.shape[0]
        cdef Alignment record = Alignment.make_empty(flen,elen)
        cdef double[:,:] problist = record.problist
        cdef int[:] ml = record.ml
        cdef double z,overallprob = 0.0
        cdef double score
        cdef int k,j
        cdef double lamb = self.alambda
        cdef bint elam = (lamb > 0.0)
        cdef double[:] wf = self._wc
        cdef bint smooth = elam and (wf.shape[0] > 1)
        cdef double div = float(flen)**float(elen)

        for k in range(elen):

            if t[k] == -1:
                continue

            z = 0.0
            for j in range(flen):

                if s[j] == -1:
                    continue

                score = (table[s[j],t[k]]*treedist[treelen,k,treepos[j]])
                z += score
                problist[k][j] = score
                if score >= problist[k][ml[k]]:
                    ml[k] = j

            ## mercer-jelinek smoothign
            if smooth:
                z = ((1-lamb)*z) +(lamb*wf[k])

            if overallprob == 0.0:
                overallprob = z
            else:
                overallprob *= z 

        record.prob = overallprob/div
        return record

    def __reduce__(self):
        # pickle implementation 
        return TreeModel,(self.source_lex,self.target_lex,self.maxlen,
                          self.encoding,self.table,self.tdistortion,
                          self._wc,self._sv,self.alambda)

### alignment object

cdef class Decoding:

    def __init__(self,tlen):
        self.tlen = tlen
        self._ts_array = np.ndarray(tlen,dtype=np.int32)
        self.prob = 1.0
        self._ts_array.fill(0)

    def __str__(self):
        return self.giza()

    @boundscheck(False)
    @wraparound(False)
    cpdef unicode giza(self):
        """print giza format

        -- assumes that source 0 is None, which does
        not get printed in the giza string (and everything
        else gets decremented by one) 

        :returns: giza formatted string
        :rtype: str
        """
        cdef unicode finalstr = u''
        cdef int i,tlen = self.tlen
        cdef int sval
        cdef int[:] alignment = self._ts_array
        
        for i in range(tlen):
            sval = alignment[i]
            if sval <= 0: continue
            sval -= 1 
            finalstr += u"%d-%d " % (i,sval)

        return finalstr

    def add_text(self,list s,list t):
        """align text input with decoding

        :param s: source text
        :type s: list(unicode)
        :param t: target text
        :type t: list(unicode)
        :returns: list of 
        :rtype: list(tuple)
        :raises: lenngth incompability 
        """
        pass

    def __richcmp__(self,other,opt):
        if opt == 2:
            return np.array_equal(self._ts_array,other._ts_array)
        elif opt == 3:
            return (not self == other)

    ### symmetrization implementations
    
        
cdef class Alignment:
    """an alignment object for representing decoder output"""
    
    def __init__(self,slen,tlen,prob,problist,ml):
        """

        :param slen: source len
        :type slen: int
        :param tlen: target len
        :type tlen: int
        :param 
        """
        self.slen = slen
        self.tlen = tlen
        self.prob = prob
        self.problist = problist
        self.ml = ml
        
    def __iter__(self):
        return iter(self.alignments())

    cdef inline int[:] _shortest_path(self,int start,double[:,:] block):
        """find the shortest alignment path given blocks using greedy search

        :param start: which point (in target) to start in the search
        :param block: points to avoid
        """
        cdef double[:,:] problist = self.problist
        cdef int tlen = self.tlen
        cdef int slen = self.slen
        cdef int[:] path = np.ndarray((tlen,),dtype=np.int32)
        cdef int i,j,nexti,size
        cdef double largest

        path[:] = -1
        
        for i in range(start,tlen):
            largest = 0.0
            
            for j in range(slen):
                if isinf(block[i][j]):
                    continue
                if problist[i][j] >= largest:
                    largest = problist[i][j]
                    path[i] = j

        return path

    cdef _k_best2(self,int k):
        """re-implemention of k-best alignments procedure"""
        cdef int slen = self.slen
        cdef int tlen = self.tlen
        cdef int i,kbest
        cdef double[:,:] problist = self.problist
        cdef double[:,:] block
        cdef int[:] ts_array,earray,path,spurpath,fullpath


        # # first best
        # kbest = self._find_best()
        # block = np.empty((tlen,slen),dtype='d')

        # ## extract next best
        # for kbest in range(1,k):
        #     ts_array = (<Decoding>kbest[-1])._ts_array
                

    
    #@boundscheck(False)
    #@wraparound(False)
    #@cdivision(True) 
    cdef list _k_best(self,int K):
        """implementation of k-best alignments using Yen's
        Kbest (loopless) shortest path algorithm (with greedy
        single shortest path subprocedure)
    
        :param K: number of alignments to compute
        :type K: int
        :return: k-best list of alignments
        :rtype: list 
        """
        cdef list B = []
        cdef list kbest
        cdef double[:,:] problist = self.problist
        cdef int slen = self.slen
        cdef int tlen = self.tlen
        cdef int i,k,j,apos,currlen,z,actual,start,cindx
        cdef double[:,:] block
        cdef int[:] ts_array,earray,path,spurpath,fullpath
        cdef bint matches
        cdef Decoding candidate
        cdef int[:] canarray
        cdef int canlen,mindx
        cdef double maxprob
        cdef Decoding maxc
        
        kbest = self._find_best()
        block = np.empty((tlen,slen),dtype='d') 
        
        for k in range(1,K):
            ts_array = (<Decoding>kbest[-1])._ts_array
            currlen = PyList_GET_SIZE(kbest)

            # # find new best paths 
            for i in range(tlen):
                block[:] = 0.0
                actual = (i - 1) if i != 0 else i
                apos = ts_array[actual]
                path = ts_array[0:actual+1]
                for j in range(currlen):
                    earray = (<Decoding>kbest[j])._ts_array
                    matches = True
                    if (i == 0) and earray[0] == apos:
                        block[0][earray[0]] = np.inf
                        continue

                    for z in range(0,i):
                        if earray[z] != path[z]:
                            matches = False
                            
                    if <bint>matches: 
                        block[i][earray[i]] = np.inf

                ## find shortest path
                spurpath = self._shortest_path(i,block)
                if spurpath[i] == -1:
                    continue
                candidate = Decoding(tlen)
                canarray = candidate._ts_array

                ## compute full path and its probability
                for j in range(tlen):
                    if j >= i and spurpath[j] >= 0:
                        cindx = spurpath[j]
                    else:
                        cindx = ts_array[j] 
                    canarray[j] = cindx
                    candidate.prob *= problist[j][cindx]

                PyList_Append(B,candidate)
            
            ## pick best candidate
            canlen = PyList_GET_SIZE(B)
            mindx = -100
            maxprob = 0.0
            
            for j in range(0,canlen):
                if <double>(B[j].prob) >= maxprob:
                    maxprob = B[j].prob
                    mindx = j 

            if mindx == -100: break 
            maxc = B[mindx] 
            kbest.append(maxc)
            while maxc in B:
                B.remove(maxc)

        return kbest
            
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
        cdef int[:] ts_array = decoding._ts_array
        
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
        if k == 1:
            return self._find_best()
        return self._k_best(k)

    @classmethod
    def make_empty(cls,slen,tlen):
        """initialize an empty alignment object

        :param slen: source input length
        :param tlen: target input length
        :returns: empty alignment instance
        :rtype: Alignment
        """
        ml = np.ndarray(tlen,dtype=np.int32)
        ml.fill(0)
        problist = np.ndarray((tlen,slen),dtype="d")
        problist.fill(0.0) 
        return cls(slen,tlen,0.0,problist,ml)


cpdef Aligner(config):
    """factory method for returning aligner

    :param config: configuration
    :returns: an (empty) aligner model
    :rtype: Model1 or Model2
    :raises: AlignerError

    >>> from zubr.Aligner import Aligner
    >>> a1 = Aligner("IBM1")
    >>> a2 = Aligner("IBM7")
    
    """
    atype = ''
    
    if isinstance(config,basestring):
        atype = config.lower()
    else:
        atype = config.modeltype.lower()

    if atype == "ibm1":
        return Model1()
    elif atype == "ibm2":
        return Model2()
    elif atype == "treemodel":
        return TreeModel()
    
    raise AlignerError(
        "aligner model not known: %s" % atype
        )

# ######################################
# ############### cli

def params():
    """main parameters for running the aligners and/or aligner experiments

    :rtype: tuple
    :returns: description of option types with name, list of options 
    """
    
    options = [
        ("--aligner","aligner",'',"str",
         "path to aligner model [default='']","Aligner"),
        ("--amax","amax",50,int,
         "maximum sentence length [default=50]","Aligner"),
        ("--source","source",'f',"str",
         "source file extension [default='f']","Aligner"),
        ("--target","target",'e',"str",
         "target file extension [default='e']","Aligner"),
        ("--ignore_oov","ignore_oov",False,"bool",
         "ignore unknown words [default=True]","Aligner"),
        ("--modeltype","modeltype",'ibm1',"str",
         "type of alignment model [default='ibm1']","Aligner"),
        ("--symmetric","symmetric",False,"bool",
         "use a symmetric aligner [default=False]","Aligner"),
        ("--word_match","word_match",False,"bool",
         "align only matching words [default=False]","Aligner"),
        ("--aalpha","aalpha",0.0,float,
         "alignment smooth prior [default=0.0]","Aligner"),
        ("--dalpha","dalpha",0.0,float,
         "distortion smooth prior [default=0.0]","Aligner"),
        ("--aiters","aiters",10,int,
         "maximum iterations for aligner [default=10]","Aligner"),
        ("--aiters2","aiters2",5,int,
         "maximum iterations for aligner model2 [default=10]","Aligner"),
        ("--aiters3","aiters3",5,int,
         "maximum iterations for tree model [default=5]","Aligner"),
        ("--atraining","atraining","","str",
         "location of aligner training data [default='']","Aligner"),
        ("--amode","amode","train","str",
         "aligner mode [default='train']","Aligner"),
        ("--aligntraining","aligntraining",False,
         "bool","align training data [default=False]","Aligner"),
        ("--aligntesting","aligntesting",False,
         "bool","align testing data [default=False]","Aligner"),
        ("--atesting","atesting","",
         "str","aligner testing data [default='']","Aligner"),
        ("--abeam","abeam",1,int,
         "beam size [default=1]","Aligner"),
        ("--emax","emax",100,int,
         "maximum size of e side input [default=100]","Aligner"),         
        ("--alambda","alambda",0,float,
         "lambda parameter for mercer-jelinek smoothing [default=0]","Aligner"),
        ("--minprob","minprob",0.0,float,
         "minimal probability to consider when aligning [default=0]","Aligner"),
        ("--remrep","remrep",False,"bool",
         "remove repeat words when testing (for Model1) [default=False]","Aligner"),
        ("--lower","lower",True,"bool",
         "map words to lowercase globally [default=True]","Aligner"),
        ("--trainstop","trainstop",10000000,int,
         "for models > 2, place where core training data stops [default=1000000]","Aligner"),
         ("--maxtree","maxtre",10,int,
         "for tree model, largest tree size [default=10]","Aligner"),
        ]

    aligner_group = {"Aligner":"Aligner settings and defaults"}
    return (aligner_group,options)

def argparser():
    """return an aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr aligner [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser 


def main(argv):
    """main execution function

    :param argv: user input or config file
    :type argv: zubr.util.ConfigAttrs or list
    """

    from zubr.util import ConfigAttrs
    
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:]) 
        logging.basicConfig(level=logging.DEBUG)
        
    aligner = Aligner(config.modeltype)
    mode = config.amode.lower() 

    try: 
        if mode == "train":
            aligner.train(config=config)
            
        elif mode == "test":
            pass

        else:
            raise AlignerError('unknown alignment mode: %s' % mode)
        
    except Exception,e:
        traceback.print_exc(file=sys.stdout)

    finally:
        if config.dump_models:
            model_out = os.path.join(config.dir,"base.model")
            aligner.dump_aligner(model_out)
        
if __name__ == "__main__":
    main(sys.argv[1:])
