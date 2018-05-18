#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementation of a symmetric aligner using aligners from Aligner.pyx 

"""

import os
import sys
import time
import logging
import shutil
import traceback
cimport numpy as np
import numpy as np
import gzip
import cPickle as pickle
import codecs
from zubr.util import ConfigAttrs
from collections import defaultdict 
from cython cimport boundscheck,wraparound,cdivision
from zubr.Alignment cimport WordModel,Alignment,Decoding
from zubr.Alignment import Aligner
from zubr.util.aligner_util import load_aligner_data,get_tree_data,load_glue_grammar
from zubr.util.alignment_util import load_aligner_data
from zubr.util.sym_align_util import *
from zubr.ZubrClass cimport ZubrSerializable

cdef class SymmetricAlignerBase(ZubrSerializable):
    """Base class for symmetric aligner"""

    def extract_phrases(self,heuristic):
        """extract phrases from symmetric alignment

        :param heuristic: particular symmetrization heuristic
        :type heuristic: str
        """
        raise NotImplementedError()
    
    cpdef int train(self,object config) except -1:
        """train a symmetric aligner

        :param config: zubr aligner configuration
        :rtype: None
        """
        raise NotImplementedError()

    cpdef SymAlign align(self,f,e,str heuristic='grow-diag'):
        """align a given input-output

        :param f: foreign input
        :param e: english input
        :param heuristic: heuristic to use when aligning
        """
        raise NotImplementedError()

    cpdef int align_dataset(self,object config) except -1:
        """Align a given dataset

        :param config: aligner configuration
        """
        raise NotImplementedError()

    cpdef int phrases_from_data(self,object config) except -1:
        """Extract (non-hierarchical) phrase rules from training data

        :param config: the main configuration 
        """
        raise NotImplementedError()

    cpdef int extract_hiero(self,object config) except -1:
        """Extract hierarchical phrase rules from data 

        :param config: the main configuration
        """
        raise NotImplementedError()
    
    cdef inline int pair_identifier(self,int eid,int fid, int flen):
        """find a unique id for a given word pair

        :param eid: e side word in integer form
        :param fid: f side word in integer form
        :param flen: the total length of f vocabulary
        :rtype: int 
        """
        return (eid*flen)+fid

    cdef SymAlign _align(self,int[:] e,int[:] f,int[:] e2,str heuristic='union'):
        """main c method for finding alignment (or decoding) given input-out

        :param e: english input
        :param f: foreign input
        :param e2: english input with null symbol (for doing e->f)
        :param heuristic: heuristic to use when making symmetrical alignment
        """
        raise NotImplementedError()

    def backup(self,wdir):
        """Alternative backup method to pickle

        :param wdir: the working directory to back up to
        """
        raise NotImplementedError

    @classmethod
    def load_backup(cls,wdir):
        """Load the backup created

        :param wdir: the working directory where files sit 
        """
        raise NotImplementedError
        
cdef class SymmetricWordModel(SymmetricAlignerBase):
    """A Symmetric word model"""

    def __init__(self,aligner1,aligner2,phrase_table={}):
        """Initializes a symmetric word base aligner model 

        :param aligner1: the first alignment model f->e
        :param aligner2: the second alignment model e->f
        :param phrase_table: the list of phrases
        """
        self.ftoe = <WordModel>aligner1
        self.etof = <WordModel>aligner2
        self.phrase_table = phrase_table

    @classmethod
    def from_config(cls,config):
        """Load a given symmetric model from a single configuration

        :param config: the main configuration
        :type config: zubr.util.config.ConfigAttrs
        :returns: SymmetricWordModel instance
        """
        model_type = Aligner(config.modeltype)
        config.sym = True

        ## setup the first aligner
        config_1 = ftoe_config(config)
        ftoe = model_type.load_data(config=config_1)

        ## setup the second aligner
        config_2 = etof_config(config)
        etof = model_type.load_data(config=config_2)

        return cls(ftoe,etof)

    ## train model

    cpdef int train(self,object config) except -1:
        """The main train function 

        :param config: the main experimental configuration
        """
        cdef WordModel ftoe = <WordModel>self.ftoe
        cdef WordModel etof = <WordModel>self.etof

        ## train f->e
        self.logger.info('Training the f->e alignment model...')
        ftoe.train()

        ## train e->f
        self.logger.info('Training the e->f alignment model...')
        etof.train()

        ## align the training data
        if config.aligntraining:
            self.logger.info('Aligning the training data...')
            self.align_dataset(config)

        ## extract phrases
        if config.extract_phrases:
            self.logger.info('Extracting phrases from data...')
            if config.extract_hiero:
                self.extract_hiero(config)
            self.phrases_from_data(config)
            # else: 
            #     self.phrases_from_data(config)

    cpdef int align_dataset(self,object config) except -1:
        """Align a dataset (currently works for training data)

        -- note: rebuilds the data 

        :param config: alignment configuration 
        :rtype config: zubr.util.config.ConfigAttrs
        :rtype: None 
        """
        cdef int i,data_size
        cdef np.ndarray f,e
        cdef str heuristic = config.aheuristic

        ## output file for alignment 
        out = sys.stdout if not config.dir \
          else codecs.open(os.path.join(config.dir,"alignment.txt"),'w')

        self.logger.info('Rebuilding the dataset for evaluation...')
        data = load_aligner_data(config)
        f = data[0]; e = data[1]
        data_size = f.shape[0]

        for i in range(data_size):
            print >>out,self._align(f[i],e[i],np.insert(e[i],0,0),heuristic=heuristic)

        ## close the output file 
        if out != sys.stdout: out.close()

    cdef SymAlign _align(self,int[:] f,int[:] e,int[:] e2,str heuristic='grow-diag'):
        """main c method for find a symmetric alignment (or decoding) given input-output pairs

        :param e: the english input/output
        :param f: the foreign input/output
        :param e2: the english input with null symbol (for doing e->f)
        :param heuristic: the heuristic to use when symmetrisizing the alignment
        :returns: symmetric alignment with heuristic applied
        """
        cdef WordModel etof = <WordModel>self.etof
        cdef WordModel ftoe = <WordModel>self.ftoe
        cdef double prob
        cdef Alignment etofa,ftoea
        cdef Decoding etofd,ftoed

        ## alignment in both directions 
        ftoea = ftoe._align(f,e)
        etofa = etof._align(e2,f[1:])
        ftoed = ftoea._find_best()[0]
        etofd = etofa._find_best()[0]
        prob = ftoea.prob

        return SymAlign(ftoed,etofd,prob,heuristic=heuristic)

    cpdef int phrases_from_data(self,object config) except -1:
        """Extract ordinary (non-hierarchical) phrases from a dataset 

        :param config: the aligner configuration 
        :type config: zubr.util.config.ConfigAttrs
        :returns: nothing 
        """
        _extract_phrases(self,config)

    cpdef int extract_hiero(self,object config) except -1:
        """Extract hierarchical phrase rules from training data 

        :param config: the main configuration 
        :type config: zubr.util.config.ConfigAttrs
        """
        try:
            _extract_hiero_phrases(self,config)
        except Exception,e:
            self.logger.error(e,exc_info=True)

    ## backup method

    def backup(self,wdir):
        """Back up the symmetric aligner model using something other than pickle 
        
        Note: relies on the backup for the ftoe and etof model 
        
        :param wdir: the working directory where the data sits 
        """
        stime = time.time()
        phrase_dir = os.path.join(wdir,"phrase_data")
        if os.path.isdir(phrase_dir):
            self.logger.info('Already backed up, skipping....')
            return

        os.mkdir(phrase_dir)
        ## back up the phrase table
        phrasep = os.path.join(phrase_dir,"phrases.gz")
        with gzip.open(phrasep,'wb') as my_path:
            pickle.dump(self.phrase_table,my_path)

        ## back up the indiviual models (relying on their backup implementation)
        self.ftoe.backup(wdir,'ftoe')
        self.etof.backup(wdir,'etof')
        
        # log the time 
        self.logger.info('Back up in %s seconds' % str(time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Loads the symmetric model backup 

        :param wdir: the working directory where associated files sit
        :returns: a SymmetricWordModel instance 
        """
        stime = time.time()
        phrase_dir = os.path.join(config.dir,"phrase_data")

        # phrase table
        pfile = os.path.join(phrase_dir,"phrases.gz")
        with gzip.open(pfile,'rb') as pd:
            phrases = pickle.load(pd)

        ## individual models
        mclass = Aligner(config.modeltype)
        ftoe = mclass.load_backup(config,tname='ftoe')
        etof = mclass.load_backup(config,tname='etof')

        ## new instance 
        instance = cls(ftoe,etof,phrases)
        instance.logger.info('Loaded in %s seconds' % (time.time()-stime))
        
        return instance 
        
    #####
    ## properties
        
    property elen:
        """Specified the number of english words in f->e aligner model"""
    
        def __get__(self):
            """Returns the number of english words in f->e

            :rtype: inr
            """
            return <int>self.ftoe.elen
        
    property flen:
        """Specified the number of foreign words in f->e aligner model"""
    
        def __get__(self):
            """ Returns the foreign word map from the f->e aligner 

            :rtype: int 
            """
            return <int>self.ftoe.flen

    property elex:
        """Specifies the english lexicon map"""

    
        def __get__(self):
            """Returns the english lexicon map 

            :rtype: dict
            """
            return <dict>self.ftoe.elex

    property flex:
        """Specifies the foreign  lexicon map"""
    
        def __get__(self):
            """Returns the foreign lexicon map 

            :rtype: dict 
            """
            return <dict>self.ftoe.flex

    property num_phrases:
        """The current number of stored phrases (if available)"""
    
        def __get__(self):
            """Returns the current size of the saved phrase table 

            -- note: the phrase table might not be saved, in which case 
            this will return 0 

            :rtype: int
            """
            cdef dict table = self.phrase_table 
            return <int>len(table)
        
    def __reduce__(self):
        ## pickle implementation
        return SymmetricWordModel,(self.ftoe,self.etof,self.phrase_table)



### C METHODS


## phrase extract (a bit messy!)
    
cdef void _extract_phrases(SymmetricWordModel model,object config):
    """Extract phrases from a symmetric word model 

    :param model: the model to extract phrases from
    :param config: the main configuration 
    """
    cdef int i,dsize,j
    cdef str heuristic = config.aheuristic
    cdef SymAlign alignment
    cdef double ptime
    cdef int max_phrase = config.max_phrase
    cdef Phrases phrases
    cdef int num_phrases
    cdef dict phrase_counts = <dict>defaultdict(int)
    cdef dict phrase_co = <dict>defaultdict(list)
    cdef dict co,phrase_table = {},pre_table = {}
    cdef list lex_phrases,elist
    cdef tuple ephrase,fphrase
    cdef set unique_ep = set()

    ptime = time.time()
    model.logger.info('Rebuilding the data again...')
    f,e,fdict,edict,_,_ = load_aligner_data(config)
    dsize = f.shape[0]

    ## extract raw phrases here 
    for i in range(dsize):
        alignment = model._align(f[i],e[i],np.insert(e[i],0,0),heuristic=heuristic)
        phrases = alignment.extract_phrases(max_phrase)
        lex_phrases = phrases.lexical_positions(f[i],e[i])
        for ephrase,fphrase in lex_phrases:
            phrase_counts[fphrase] += 1
            phrase_co[fphrase].append(ephrase)

    ## normalize phrase counts
    for fphrase,elist in phrase_co.items():
        co = <dict>defaultdict(int)
        for ephrase in elist:
            co[ephrase] += 1
            unique_ep.add(ephrase)
        for ephrase in set(elist):
            if fphrase not in pre_table:
                pre_table[fphrase] = {}
            pre_table[fphrase][ephrase] = float(co[ephrase])/float(phrase_counts[fphrase])

    ## put into a flat list
    for fphrase,eitems in pre_table.items():
        for item in eitems:
            # reversed order..
            phrase_table[(fphrase,item)] = pre_table[fphrase][item]

    ## model info about the extraction
    model.logger.info('extracted phrases (%d foreign, %d english) in %s seconds' %\
      (len(phrase_counts),len(unique_ep),str(time.time()-ptime)))

    ## save phrases to model
    if config.save_phrases:
        model.phrase_table = phrase_table

    ## print phrases out:
    if config.print_table or config.backup:
        frev = {o[1]:o[0] for o in fdict.iteritems()}
        erev = {o[1]:o[0] for o in edict.iteritems()}
        print_phrase_table(frev,erev,phrase_table,config)


## hiero extract (very messy!!)

cdef int _extract_hiero_phrases(SymmetricWordModel model,object config) except -1:
    """Extract hierarchical rules from a symmetric model with tree information 


    :param model: the model to use to derive phrases 
    :param config: the main configuration 
    """
    cdef int i,dsize,j,k,q
    cdef str heuristic = config.aheuristic
    cdef SymAlign alignment
    cdef double ptime
    cdef int max_phrase = config.max_phrase
    cdef Phrases phrases
    cdef int num_phrases
    cdef dict lookup_table
    cdef list lex_phrases,elist
    cdef tuple ephrase,fphrase
    cdef dict hiero_rules = <dict>defaultdict(int)
    cdef int num_trees,flen,elen
    cdef int[:] ftreepos
    cdef int size,p1,p2,e1,e2,tp1,tp2
    cdef list echart,fchart,ep,fp,eseq,fseq
    cdef int[:,:] local_phrases
    cdef int[:] e_aligned,f_aligned
    cdef tuple tuple1,finfo
    cdef unicode lhs,tlhs1
    cdef int spansize,start,end,mid
    cdef int nstart,nend,nfstart,nfend
    cdef list fspan,espan
    cdef int fstart,fend
    cdef int fs1,fs2,fe1,fe2,g1,g2

    model.logger.info('Rebuilding the data again...')
    f,e,fdict,edict,_,_ = load_aligner_data(config)
    dsize = f.shape[0]

    ## get the tree information for the data 
    model.logger.info('Reading tree file...')
    tree_pos = get_tree_data(config,tset='train')
    model.logger.info('datasize=%d, tree size=%d' % (e.shape[0],tree_pos.shape[0]))

    rule_map = load_glue_grammar(config)
    num_trees = tree_pos.shape[0]

    ## start the counter
    ptime = time.time()

    for i in range(num_trees):
            
        ## alignment
        alignment = model._align(f[i],e[i],np.insert(e[i],0,0),heuristic=heuristic)
        ## phrase positions
        phrases = alignment.extract_phrases(max_phrase)
        size = phrases.phrases.shape[0]

        ## extract phrase rule from examples with a tree structure

        #if True:
        ftreepos = tree_pos[i][:-1]
        flen = f[i].shape[0]
        elen = e[i].shape[0]
        
        assert ftreepos.shape[0] == flen, 'wrong length at %d, tree=%d, flen=%d' % (i,ftreepos.shape[0],flen)
        echart = [[u'' for _ in range(elen+1)] for _ in range(elen)]
        fchart = [[u'' for _ in range(flen+1)] for _ in range(flen)]
        lookup_table = {}

        local_phrases = phrases.phrases
        e_aligned = np.zeros((elen,),dtype=np.int32)
        f_aligned = np.zeros((flen,),dtype=np.int32)

        ## first find smallest phrases and unaligned points
        
        for j in range(size):

            e1 = local_phrases[j][0]
            e2 = local_phrases[j][1]
            p1 = local_phrases[j][2]
            p2 = local_phrases[j][3]

            tp1 = ftreepos[p1]
            tp2 = ftreepos[p2]
            tuple1 = (str(tp1),str(tp2))
            tlhs1 = rule_map.get(tuple([str(tp1)]),u'')
                    
            ## check alignment points 
            for k in range(e1,e2+1): e_aligned[k] = 1
            for k in range(p1,p2+1): f_aligned[k] = 1
                        
            ## smallest spans 
            if e1 == e2:
                ## single tree position
                #lhs = rule_map.get(tuple([str(tp1)]),u'')
                lhs = tlhs1

                if tp1 == tp2 and lhs:
                    echart[e1][e2+1] = lhs
                    fspan = [w for w in f[i][p1:p2+1]]
                    espan = [w for w in e[i][e1:e2+1]] 
                    hiero_rules[(lhs,tuple(espan),tuple(fspan))] += 1
                    lookup_table[(e1,e2+1)] = (p1,p2+1)
                            
                ## multiple tree positions
                    
                elif tuple1 in rule_map and (p2-p1 <= 3):
                    lhs = rule_map[tuple1]
                    
                    ## make sure fspan is complete (doesnt contain fragments of trees on either side)
                    ## left
                    if p1 >= 1 and ftreepos[p1-1] == ftreepos[p1]: p1 += 1
                            
                    #right
                    if ftreepos[p2-2] == ftreepos[p2-1]: p2 -= 1
                    ## right
                    #if p2 == p1: continue 
                    fspan = [w for w in f[i][p1:p2+1]]
                    if not fspan: continue
                            
                    espan = [w for w in e[i][e1:e2+1]]
                    lookup_table[(e1,e2+1)] = (p1,p2+1)
                    echart[e1][e2+1] = lhs
                    hiero_rules[(lhs,tuple(espan),tuple(fspan))] += 1

                ## can be fixed with one shift to the right
                    
                elif not p2-1 <= p1 and ftreepos[p2-1] == tp1 and lhs:
                    espan = [w for w in e[i][e1:e2+1]]
                    lookup_table[(e1,e2+1)] = (p1,p2)
                    echart[e1][e2+1] = lhs
                    fspan = [w for w in f[i][p1:p2]]
                    hiero_rules[(lhs,tuple(espan),tuple(fspan))] += 1
                        
            ## longer spans with abstract lhs 
            elif (e2-e1 <= 3) and (tp1 == tp2) and tlhs1:
                echart[e1][e2+1] = tlhs1
                fspan = [w for w in f[i][p1:p2+1]]
                espan = [w for w in e[i][e1:e2+1]]
                hiero_rules[(tlhs1,tuple(espan),tuple(fspan))] += 1
                lookup_table[(e1,e2+1)] = (p1,p2+1)

            ## longer spans with rhs rules
            elif (e2-e1 <= 3) and tuple1 in rule_map and (p2-p1 <= 3):
                lhs = rule_map[tuple1]
                    
                ## left 
                if p1 >= 1 and ftreepos[p1-1] == ftreepos[p1]: p1 += 1
                    
                ## right
                if ftreepos[p2-2] == ftreepos[p2-1]: p2 -= 1
                fspan = [w for w in f[i][p1:p2+1]]
                if not fspan: continue

                espan = [w for w in e[i][e1:e2+1]]
                lookup_table[(e1,e2+1)] = (p1,p2+1)
                echart[e1][e2+1] = lhs
                hiero_rules[(lhs,tuple(espan),tuple(fspan))] += 1

                ## longer spans with


        ## now find subspans
        for spansize in range(2,elen+1):
            for start in range(elen - spansize+1):
                end = start + spansize

                ### GLUE on aligned items phrases inside unary rules
                if (start,end) not in lookup_table:
                    nstart = start
                    nend   = end
                        
                    if e_aligned[start] < 1 or e_aligned[end-1] < 1:

                        ## find start 
                        while True:
                            if nstart+1 >= end or (nstart-start) >= 3:
                                nstart = start
                                break

                            if e_aligned[nstart] == 1: 
                                break
                            nstart += 1

                        ## find end
                        while True:
                            if nend-1 <= start or (end-nend) >= 3:
                                nend = end
                                break
                            if e_aligned[nend-1] == 1:
                                break
                            nend -= 1

                    tag = echart[nstart][nend]

                    if tag:
                        finfo = lookup_table[(nstart,nend)]
                        ## move left to right to find extract words
                        fstart = finfo[0]
                        fend = finfo[1]
                        #ftree = ftreepos[<int>finfo[0]]
                        ftree = ftreepos[fstart]

                        nfstart = fstart
                        nfend = fend 
                                                                                    
                        while True:
                            if nfstart <= 1 or ftreepos[nfstart-1] != ftree or (fstart-nfstart) >= 3:
                                break
                            nfstart -= 1

                        while True:

                            if nfend+1 >= flen or ftreepos[nfend+1] != ftree or (nfend-fend) >= 3:
                                break
                            nfend += 1

                        espan = [e[i][w] for w in range(start,nstart)]+[tag]+\
                          [e[i][w] for w in range(nend,end)]


                        fspan = [f[i][w] for w in range(nfstart,fstart)]+[tag]+\
                          [f[i][w] for w in range(fend,nfend) if w < flen]

                        hiero_rules[(tag,tuple(espan),tuple(fspan))] += 1
                        echart[start][end] = tag
                        lookup_table[(start,end)] = (nfstart,nfend)

                        #continue 
                                                        
                    # ## find the middle points
                for mid in range(start+1,end):
                    span1 = echart[start][mid]
                    span2 = echart[mid][end]

                    if (not span1 or not span2) or (span1,span2) not in rule_map:
                        continue

                    lhs = rule_map[(span1,span2)]
                    
                    ## find all gaps inside these two spans
                        
                    nstart = start
                    nend = end
                        
                    fs1,fe1 = lookup_table[(start,mid)]
                    fs2,fe2 = lookup_table[(mid,end)]
                    if (fs1,fe1) == (fs2,fe2): continue
                    fstart = fs1 if fs1 < fs2 else fs2
                    fend = fe1 if fs1 > fs2 else fe2

                    nfend = fend
                    nfstart = fstart
                    ## see if there is a gap
                    
                    if (fs1 == fstart):
                        if fs2 < fe1: continue
                        g1 = fe1
                        g2 = fs2

                        
                    elif (fs2 == fstart):
                        if fs1 < fe2: continue
                        g1 = fe2
                        g2 = fs1

                    if g2 - g1 > 3:
                        continue

                        ## added to chart 

                        ### move e 
                        
                    ## move right 
                    while True:
                        if nstart == 0:
                            break

                        if e_aligned[nstart-1] == 1 or (start-nstart) > 3:
                            break

                        nstart -= 1

                    while True:
                        if nend >= elen:
                            break

                        if e_aligned[nend] == 1 or (nend - end ) > 3:
                            break

                        nend += 1
                            
                        ## move f (check that it follows tree patterns)

                        ## move right 
                    while True:
                        if nfstart == 1:
                            break

                        #if f_aligned[nfstart-1] == 1 or (start-nfstart) > 3:
                        if f_aligned[nfstart-1] == 1 or (fstart-nfstart) > 3:
                            break

                        nfstart -= 1

                    ## move left

                    while True:

                        if nfend >= flen:
                            break

                        if f_aligned[nfend] == 1 or (nfend - fend) > 3:
                            break

                        nfend += 1

                    echart[nstart][nend] = lhs
                    lookup_table[(nstart,nend)] = (nfstart,nfend)

                    if (nend - nstart) <= 3 and (nfend - nfstart) <= 3:

                        ## check that the trees are the right span
                        espan = [e[i][w] for w in range(nstart,nend)]
                        fspan = [f[i][w] for w in range(nfstart,nfend)]
                        hiero_rules[(lhs,tuple(espan),tuple(fspan))] += 1


                    espan = [e[i][w] for w in range(nstart,start)]+\
                      ["%s_1" % span1,"%s_2" % span2]+[e[i][w] for w in range(end,nend)]


                    if fstart == fs1: 
                        fspan = [f[i][w] for w in range(nfstart,fstart)]+["%s_1" % span1]+\
                          [f[i][w] for w in range(g1,g2)]+["%s_2" % span2]+\
                          [f[i][w] for w in range(fend,nfend)]

                        hiero_rules[(lhs,tuple(espan),tuple(["%s_1" % span1,"%s_2" % span2]))] += 1

                    else:
                        fspan = [f[i][w] for w in range(nfstart,fstart)]+["%s_2" % span2]+\
                          [f[i][w] for w in range(g1,g2)]+["%s_1" % span1]+\
                          [f[i][w] for w in range(fend,nfend)]

                        hiero_rules[(lhs,tuple(espan),tuple(["%s_2" % span1,"%s_1" % span2]))] += 1

                    #hiero_rules.add((lhs,tuple(espan),tuple(fspan)))
                    hiero_rules[(lhs,tuple(espan),tuple(fspan))] += 1

                    ### additional rules (depart a bit from standard hiero rules)
                    if mid-start <= 2: 
                        espan = [e[i][w] for w in range(start,mid)]+["%s_2" % span2]
                        hiero_rules[(lhs,tuple(espan),tuple(fspan))] += 1

                        ## put in the rule without rhs gaps 
                        if fstart == 1:
                            hiero_rules[(lhs,tuple(espan),tuple(["%s_1" % span1, "%s_2" % span2]))] += 1
                        else:
                            hiero_rules[(lhs,tuple(espan),tuple(["%s_2" % span2,"%s_1" % span1, ]))] += 1

                    if end-mid <= 2:
                        espan = ["%s_1" % span1]+[e[i][w] for w in range(mid,end)]
                        hiero_rules[(lhs,tuple(espan),tuple(fspan))] += 1

                        ## put in the rule without rhs gaps 
                        if fstart == 1:
                            hiero_rules[(lhs,tuple(espan),tuple(["%s_1" % span1, "%s_2" % span2]))] += 1
                        else:
                            hiero_rules[(lhs,tuple(espan),tuple(["%s_2" % span2,"%s_1" % span1, ]))] += 1
                                
    model.logger.info('finished extraction in %s seconds' % str(time.time()-ptime))
    
    frev = {o[1]:o[0] for o in fdict.iteritems()}
    erev = {o[1]:o[0] for o in edict.iteritems()}
    grammar_out = os.path.join(config.dir,"hiero_rules.txt")

    ## print the grammar 
    with codecs.open(grammar_out,'w',encoding='utf-8') as gram:
        for item,k in hiero_rules.items():
            espan = [erev[op] if not isinstance(op,basestring) else "[%s]" % op for op in item[1]]
            fspan = [frev[op] if not isinstance(op,basestring) else "[%s]" % op for op in item[2]]
            out = "%s\t%s ||| %s\tcount=%s" % (item[0],' '.join(espan),' '.join(fspan),str(k))
            print>>gram,out    
    

    
## symmetric alignment

cdef class SymAlign:

    def __init__(self,Decoding ftoe,Decoding etof,prob,heuristic='union'):
        """
        
        :param etof: etof viterbi alignment
        :param ftoe: ftoe viterbi alignment
        :param heuristic: heurstic to use when making symmetric
        :raises: ValueError
        """
        impl_heuristics = {'union':0,'intersect':1,'grow-diag':2,'grow-diag-final':3,
                           'grow-diag-final-and':4,'grow-diag-final-final':5}
        if heuristic not in impl_heuristics:
            raise ValueError('unknown heuristic: %s' % heuristic)

        self.heuristic = impl_heuristics[heuristic]
        self.elen = ftoe.tlen
        self.flen = etof.tlen
        
        #flen+1 (include null item)
        self.alignment = np.ndarray((self.elen,self.flen+1),dtype='d')
        self.union_alignment = np.empty_like(self.alignment)
        self.alignment.fill(np.inf)
        self.union_alignment.fill(np.inf)

        ## symmetrisize the alignment 
        self.make_symmetric(ftoe.positions,etof.positions)

        ##probability
        self.prob = prob                
        

    @wraparound(False)
    @boundscheck(False)
    cdef void make_symmetric(self,int[:] ftoe,int[:] etof):
        """Symmetrisize both alignment using heuristics


        :param etof: e->f viterbi alignment, or p(e | f)
        :param ftoe: f->e viterbi alignment, or p(f | e)
        """
        cdef double[:,:] alignment = self.alignment
        cdef double[:,:] union_a = self.union_alignment
        cdef int heuristic = self.heuristic
        cdef int elen = self.elen
        cdef int flen = self.flen

        ## union 
        if heuristic == 0:
            _union(ftoe,etof,alignment,union_a,flen,elen)
        ## intersection
        elif heuristic == 1:
            _intersect(ftoe,etof,alignment,union_a,flen,elen)
        ## grow-diag
        elif heuristic == 2:
            _grow(ftoe,etof,alignment,union_a,flen,elen)
        ## grow-diag-final
        elif heuristic == 3:
            _grow_diag_final(ftoe,etof,alignment,union_a,flen,elen)
        ## grow-diag-final-and
        elif heuristic == 4:
            _grow_diag_final_and(ftoe,etof,alignment,union_a,flen,elen)
        # elif heuristic == 5:
        #     _grow_diag_final_final(ftoe,etof,alignment,union_a,flen,elen)        
        else:
            raise NotImplementedError()

    cpdef Phrases phrases(self,int max_len):
        """python method for extracting phrases

        :param max_len: maximum phrase length
        :type max_len: int
        :rtype: list
        """
        return self.extract_phrases(max_len)
    
    @wraparound(False)
    @boundscheck(False) 
    cdef Phrases extract_phrases(self,int max_len):
        """extract phrases from a given alignment

        -- note : assumes a null sumbol of the f side
        
        :param max_len: maximum size of phrases to extract
        """
        cdef int i,j,k
        cdef double[:,:] alignment = self.alignment
        cdef int elen = self.elen
        cdef int flen = self.flen+1 
        cdef int[:,:] alignment_points
        cdef double[:] e_aligned
        cdef double[:] f_aligned
        cdef Phrases phrase_obj = Phrases((elen**2)*(flen**2))
        cdef int[:,:] phrases = phrase_obj.phrases
        cdef int num_point = 0,total_phrases = 0
        cdef e,f,fstart,fend

        alignment_points = np.ndarray((elen*flen,2),dtype=np.int32)
        e_aligned = np.ndarray((elen,),dtype='d')
        f_aligned = np.ndarray((flen,),dtype='d')
        alignment_points[:,:] = -1
        e_aligned[:] = np.inf
        f_aligned[:] = np.inf
    
        ## fill alignment points (might want to change to an inline function)
        for i in range(elen):
            for j in range(0,flen):
                if isfinite(alignment[i,j]):
                    alignment_points[num_point][0] = i
                    alignment_points[num_point][1] = j
                    e_aligned[i] = 1.0
                    f_aligned[j] = 1.0
                    num_point += 1

        alignment_points = alignment_points[:num_point]

        ## now extract phrases
        for i in range(elen):
            for j in range(i,elen):
                ## exclude phrases above size
                if (j - i) > max_len:
                    continue 
                fstart = flen-1
                fend = 0
                for k in range(num_point):
                    e = alignment_points[k][0]
                    f = alignment_points[k][1]
                    if i <= e <= j:
                        fstart = is_min(f,fstart)
                        fend = is_max(f,fend)

                # ## extract phrases
                total_phrases = find_phrases(alignment_points,num_point,fstart,fend,i,j,
                                             flen,elen,e_aligned,f_aligned,phrases,total_phrases,
                                             max_len)

        ## limit to total_phrases
        phrase_obj.reduce_phrase(total_phrases)        
        return phrase_obj

    cpdef unicode giza(self):
        """print giza format

        -- assumes that source 0 is None

        :returns: giza formatted unicode string
        :rtype: unicode
        """
        cdef unicode finalstr = u''
        cdef double[:,:] alignment = self.alignment
        cdef int i,j
        cdef double sval
        cdef int elen = self.elen
        cdef int flen = self.flen+1

        for i in range(elen):
            for j in range(1,flen):
                sval = alignment[i][j]
                if isfinite(sval):
                   finalstr += u"%d-%d " % (i,j-1)
        return finalstr

    ## print alignment as string
    def __str__(self):
        return self.giza()

## global phrase extractor ()

@boundscheck(False)
@cdivision(False)
cdef np.ndarray from_giza(unicode giza_string,int elen,int flen,int estart,int eend,bint reverse=False):
    """create phrases from giza string representation

    -- note: giza representation has already been symmetrized

    :param giza_string: the alignment in giza format e->f
    """
    cdef list alignments = giza_string.split()
    cdef int apoints = len(alignments)
    cdef double[:,:] alignment = np.ndarray((elen,flen+1),dtype='d')
    cdef int i,ep,fp
    cdef Phrases phrases

    alignment[:,:] = np.inf

    ## create alignment array first
    for i in range(apoints):
        sl = alignments[i].split('-')
        if reverse:
            ep = int(sl[1])
            fp = int(sl[0])
        else:
            ep = int(sl[0])
            fp = int(sl[1])
        ## add 1 to fp, extractor assumes empty string on fside
        alignment[ep][fp+1] = 1.0

    phrases = extract_phrases(alignment,elen,flen+1,estart,eend)
    return phrases.phrases

#@boundscheck(False)
@cdivision(True)
cdef Phrases extract_phrases(double[:,:] alignment,int elen,int flen,int estart,int eend):
        """extract phrases from a given alignment given e begin and end positions

        -- note : assumes a null sumbol of the f side
        
        :param max_len: maximum size of phrases to extract
        """
        cdef int i,j,k
        cdef int[:,:] alignment_points
        cdef double[:] e_aligned
        cdef double[:] f_aligned
        cdef Phrases phrase_obj = Phrases((elen**2)*(flen**2))
        cdef int[:,:] phrases = phrase_obj.phrases
        cdef int num_point = 0,total_phrases = 0
        cdef e,f,fstart,fend
        cdef int max_len = 12

        alignment_points = np.ndarray((elen*flen,2),dtype=np.int32)
        e_aligned = np.ndarray((elen,),dtype='d')
        f_aligned = np.ndarray((flen,),dtype='d')
        alignment_points[:,:] = -1
        e_aligned[:] = np.inf
        f_aligned[:] = np.inf
    
        ## fill alignment points (might want to change to an inline function)
        for i in range(elen):
            ## skip over stuff we don't need
            for j in range(0,flen):
                if isfinite(alignment[i,j]):
                    alignment_points[num_point][0] = i
                    alignment_points[num_point][1] = j
                    e_aligned[i] = 1.0
                    f_aligned[j] = 1.0
                    num_point += 1

        alignment_points = alignment_points[:num_point]

        fstart = flen-1
        fend = 0
        for k in range(num_point):
            e = alignment_points[k][0]
            f = alignment_points[k][1]
            if estart <= e <= eend:
                fstart = is_min(f,fstart)
                fend = is_max(f,fend)

        total_phrases = find_phrases(alignment_points,num_point,fstart,fend,estart,eend,
                                     flen,elen,e_aligned,f_aligned,phrases,total_phrases,
                                     max_len)

        ## assumes the english start and end positions
        
        ## limit to total_phrases
        phrase_obj.reduce_phrase(total_phrases)        
        return phrase_obj

        
## phrase class

## phrase class

cdef class Phrases:
    """Classes for representing phrases"""

    def __init__(self,total):
        """

        :param total: total number of possible phrases
        :type total: int
        """
        self.phrases = np.ndarray((total,4),dtype=np.int32)
        self.phrases.fill(-1)

    cpdef list phrase_list(self,e,f):
        """return string phrases

        -- note : assumes that null symbol is gone on f side
        
        :param e: english input
        :type e: str
        :param f: foreign input
        :type f: str
        """
        cdef int i
        cdef int[:,:] phrases = self.phrases
        cdef int size = phrases.shape[0]
        cdef list phrase_list = []
        cdef str en_str,f_str
        cdef list en_split = e.split()
        cdef list f_split = f.split()

        for i in range(size):
            en_str = ' '.join(en_split[phrases[i][0]:phrases[i][1]+1])
            f_str = ' '.join(f_split[phrases[i][2]-1:phrases[i][3]])
            phrase_list.append((en_str,f_str))
            
        return phrase_list

    cdef void reduce_phrase(self,int size):
        """change dimension of phrases"""
        self.phrases = self.phrases[:size]
        
    cpdef list lexical_positions(self,int[:] f,int[:] e):
        """extract positions from phrases matrix

        :param f: foreign language input
        :param e: english language input
        """
        cdef int i
        cdef int[:,:] phrases = self.phrases
        cdef int size = phrases.shape[0]
        cdef list phrase_list = [],ep,fp
        cdef int w

        for i in range(size):
             ep = [w for w in e[phrases[i][0]:phrases[i][1]+1]] #.tolist()
             fp = [w for w in f[phrases[i][2]:phrases[i][3]+1]] # .tolist()
             if ep and fp:
                phrase_list.append((tuple(ep),tuple(fp)))

        return phrase_list
    
## CLI information

def params():
    """main parameters for running the aligners and/or aligner experiments

    :rtype: tuple
    :returns: description of option types for symmetric aligner
    """
    #from zubr.Aligner import params
    from zubr.Alignment import params
    aligner_group,aligner_param = params()
    aligner_group["SymmetricAligner"] = "Settings for symmetric aligner"

    options = [
        ("--extract_phrases","extract_phrases",True,"bool",
         "extract and count phrases after training [default=True]","SymmetricAligner"),
        ("--aheuristic","aheuristic",'grow-diag',"str",
         "phrase heuristic to use [default='grow-diag']","SymmetricAligner"),
        ("--max_phrase","max_phrase",7,"int",
         "maximum phrase size [default=3]","SymmetricAligner"),
        ("--train_type","train_type","EM","int",
         "method for training model [default='EM']","SymmetricAligner"),
        ("--print_table","print_table",False,"bool",
         "print the phrase table [default=False]","SymmetricAligner"),
        ("--extract_hiero","extract_hiero",False,"bool",
         "extract hiero style rules [default=False]","SymmetricAligner"),
        ("--save_phrases","save_phrases",True,"bool",
         "save phrases to model [default=True]","SymmetricAligner"),         
    ]

    options += aligner_param
    return (aligner_group,options)

def argparser():
    """return an symmetric aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr symmetricalignment [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

def main(argv):
    """The main execution call point 

    :param argv: cli input or pipeline configuration 
    :rtype argv: basestring or zubr.util.config.ConfigAttrs
    :rtype: None 
    """

    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv)
        logging.basicConfig(level=logging.DEBUG)

    try:
        
        ## train for a single corpus
        if isinstance(config.atraining,basestring):
            aligner = SymmetricWordModel.from_config(config)
            aligner.train(config)

        ## train individual models for a set of corpora 
        else:
            datasets = config.atraining
            for dset in config.atraining:
                config.atraining = dset
                ## new aligner
                aligner = SymmetricWordModel.from_config(config)
                aligner.train(config)
                ## organize directory
                dataset_name = dset.split('/')[-1]
                # rename the alignment directory (and remove alignment2)
                alignment1 = os.path.join(config.dir,'alignment')
                alignment2 = os.path.join(config.dir,'alignment2')
                viterbi = os.path.join(config.dir,'alignment.txt')
                new_dir = os.path.join(config.dir,dataset_name)
                new_viterbi = os.path.join(config.dir,'%s_alignment.txt' % dataset_name)

                ## new alignment directory
                shutil.copytree(alignment1,new_dir)
                shutil.rmtree(alignment1)
                shutil.rmtree(alignment2)
                    
                ## copy alignment file 
                shutil.copy(viterbi,new_viterbi)
                os.remove(viterbi)

                ## copy phrase_table (if it exists)
                pt = os.path.join(config.dir,'phrase_table.txt')
                npt = os.path.join(config.dir,'phrase_table_%s.txt' % dataset_name)
                    
                if os.path.isfile(pt):
                    shutil.copy(pt,npt)
                    os.remove(pt)

            config.atraining = datasets
        
    ## exception 
    except Exception,e:
        try: aligner.logger.error(e,exc_info=True)
        except: pass 
        traceback.print_exc(file=sys.stdout)
    finally:
        if config.dump_models:
            try:
                model_out = os.path.join(config.dir,"base.model")
                aligner.dump(model_out)
            except:
                pass

        elif config.pipeline_backup:
            try:
                aligner.backup(config.dir)
            except:
                pass

    # aligner = SymmetricAligner(config)
    # mode = config.amode.lower()
    # config.sym = True
    
    # try: 
    #     if mode == "train":
    #         if isinstance(config.atraining,basestring):
    #             aligner.train(config)
    #         ## train a set of corpora
    #         else:
    #             datasets = config.atraining

    #             ## train model on each dataset
    #             for dset in config.atraining:
    #                 config.atraining = dset
    #                 aligner.train(config)

    #                 ## organize directory
    #                 dataset_name = dset.split('/')[-1]

    #                 ## rename the alignment directory (and remove alignment2)
    #                 alignment1 = os.path.join(config.dir,'alignment')
    #                 alignment2 = os.path.join(config.dir,'alignment2')
    #                 viterbi = os.path.join(config.dir,'alignment.txt')
    #                 new_dir = os.path.join(config.dir,dataset_name)
    #                 new_viterbi = os.path.join(config.dir,'%s_alignment.txt' % dataset_name)

    #                 ## new alignment directory
    #                 shutil.copytree(alignment1,new_dir)
    #                 shutil.rmtree(alignment1)
    #                 shutil.rmtree(alignment2)

    #                 ## copy alignment file 
    #                 shutil.copy(viterbi,new_viterbi)
    #                 os.remove(viterbi)

    #                 ## copy phrase_table (if it exists)
    #                 pt = os.path.join(config.dir,'phrase_table.txt')
    #                 npt = os.path.join(config.dir,'phrase_table_%s.txt' % dataset_name)
                    
    #                 if os.path.isfile(pt):
    #                     shutil.copy(pt,npt)
    #                     os.remove(pt)

    #             config.atraining = datasets
                         
    #     elif mode == "test":
    #         pass 
    #     else:
    #         raise ValueError('unknown alignment mode: %s' % mode)
        
    # except Exception,e:
    #     traceback.print_exc(file=sys.stdout)

    # finally:
    #     if config.dump_models:
    #         model_out = os.path.join(config.dir,"base.model")
    #         aligner.dump_aligner(model_out)

## min max functions

cdef inline int is_min(int f,int fstart):
    if f < fstart:
        return f
    return fstart

cdef inline int is_max(int f,int fend):
    if f > fend:
        return f
    return fend

## implementation of symmetrizations

## find phrases

cdef inline int find_phrases(int[:,:] points,int num_points,int fstart,int fend,
                             int estart,int eend,int flen,int elen,double[:] e_aligned,
                             double[:] f_aligned,int[:,:] phrases,int num_phrases,
                             int max_len):
    """the main method for finding phrases"""
    cdef int i
    cdef int ep,fp,fe,fs
    cdef bint good_pos = True
    cdef int added = num_phrases

    if fend < 0:
        good_pos = False 
    else:
        for i in range(num_points):
            ep = points[i][0]
            fp = points[i][1]
            if fstart <= fp <= fend and (ep < estart or ep > eend):
                good_pos = False
                    
    ## if position is consistent
    if good_pos:
        fs = fstart
        while True:
            fe = fend

            while True:

                if (fe - fs) > max_len:
                    break
                
                if fe >= fs and fs > 0:
                    ## phrases here
                    #phrases[added]
                    phrases[added][0] = estart
                    phrases[added][1] = eend
                    phrases[added][2] = fs
                    phrases[added][3] = fe
                    added += 1
                fe += 1
                if fe >= flen or isfinite(f_aligned[fe]):
                    break
            fs -= 1
            if fs < 1 or isfinite(f_aligned[fs]):
                break
    return added

## symmetrizations

## swap 

cdef inline void _swap(int[:] etof,double[:,:] reverse,
                       double[:,:] union_a,int flen):
        """swap the etof alignment for use in intersection and other heuristics

        :param etof: etof viterbi alignment
        :param reverse: the reverse or swap alignment to fill
        :param flen: the length of f
        """
        cdef int i,eid

        for i in range(flen):
            eid = etof[i]-1
            if eid >= 0:
                reverse[eid][i+1] = 1.0
                union_a[eid][i+1] = 1.0

## intersect

cdef inline void _intersect(int[:] ftoe,int[:] etof,double[:,:] alignment,
                            double[:,:] union_a,int flen,int elen):
        """intersect the alignment for f->e and e->f

        :param ftoe: f->e viterbi alignment, or p(e | f)
        :param etof: e->f viterbi alignment, or p(f | e)
        :param alignment: the main alignment matrix to fill
        :param flen: length of f input
        :param elen: length of e input
        """
        cdef int i
        cdef int etranslation

        _swap(etof,alignment,union_a,flen)

        ## fill alingment with things in both e->f and f->e
        for i in range(elen):
            etranslation = ftoe[i]
            for j in range(flen+1):
                ## change if not consistent with f->e
                if isfinite(alignment[i][j]) and j != etranslation:
                    alignment[i][j] = np.inf
                elif j == etranslation:
                    union_a[i][j] = 1.0
                    

cdef inline void _union(int[:] ftoe,int[:] etof,double[:,:] alignment,
                        double[:,:] union_a, int flen,int elen):
        """Takes the union of both alignment

        -- note, when working with the f->e side, the e-side contains
        the null symbol 

        :param etof: e->f viterbi alignment, or p(f | e)
        :param ftoe: f->e viterbi alignment, or p(e | f)
        """
        cdef int i,j,eid

        ## fill alignment matrix with e->f
        _swap(etof,alignment,union_a,flen)

        ## fill alignment matrix with f->e         
        for i in range(elen):
            alignment[i][ftoe[i]] = 1.0
            union_a[i][ftoe[i]] = 1.0


# neighboring = np.array([[-1,0],[0,-1],[1,0],[0,1],
#                         [-1,-1],[-1,1],[1,-1],[1,1]])


cdef inline void _grow(int[:] ftoe,int[:] etof,double[:,:] alignment,
                       double[:,:] union_a,int flen,int elen):
    cdef int etranslation
    cdef int i,j,k
    cdef list neighbors = [[-1,0],[0,-1],[1,0],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
    cdef int num_neighbors = len(neighbors)
    cdef int left,right
    cdef double[:] etrans = np.ndarray((elen,),dtype='d')
    cdef double[:] ftrans = np.ndarray((flen+1,),dtype='d')
    cdef bint has_new

    etrans[:] = np.inf
    ftrans[:] = np.inf

    ## fill alignment matrix with e->f
    _swap(etof,alignment,union_a,flen)

    ## intersect the main alignment 
    for i in range(elen):
        etranslation = ftoe[i]
        for j in range(flen+1):
            ## change if not consistent with f->e
            if isfinite(alignment[i][j]) and j != etranslation:
                alignment[i][j] = np.inf
            elif j == etranslation:
                union_a[i][j] = 1.0

            ## make has alignment set
            if isfinite(alignment[i][j]):
                ftrans[j] = 1.0
                etrans[i] = 1.0
            

    while True:
        has_new = False
        
        for i in range(elen):
            for j in range(flen+1):
                if isfinite(alignment[i][j]):
                    ## look at surrounding points
                    for k in range(num_neighbors):
                        left  = neighbors[k][0]
                        right = neighbors[k][1]

                        ## neighbors are not meaningful 
                        if (i+left < 0) or (j +right < 0) or (i+left >= elen) or (j + right >= flen+1): continue
                        if (isinf(etrans[i+left]) or isinf(ftrans[j+right])) and isfinite(union_a[i+left][j+right]):
                            alignment[i+left][j+right] = 1.0
                            etrans[i+left] = 1.0
                            ftrans[j+right] = 1.0
                            has_new = True

        if not has_new:
            break 
    

cdef inline void _grow_diag_final(int[:] ftoe,int[:] etof,double[:,:] alignment,
                                  double[:,:] union_a,int flen,int elen):
    """starting point for starting grow heuristics"""
    cdef int etranslation
    cdef int i,j,k
    cdef list neighbors = [[-1,0],[0,-1],[1,0],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
    cdef int num_neighbors = len(neighbors)
    cdef int left,right
    cdef double[:] etrans = np.ndarray((elen,),dtype='d')
    cdef double[:] ftrans = np.ndarray((flen+1,),dtype='d')
    cdef bint has_new

    etrans[:] = np.inf
    ftrans[:] = np.inf

    ## fill alignment matrix with e->f
    _swap(etof,alignment,union_a,flen)

    ## intersect the main alignment 
    for i in range(elen):
        etranslation = ftoe[i]
        for j in range(flen+1):
            ## change if not consistent with f->e
            if isfinite(alignment[i][j]) and j != etranslation:
                alignment[i][j] = np.inf
            elif j == etranslation:
                union_a[i][j] = 1.0

            ## make has alignment set
            if isfinite(alignment[i][j]):
                ftrans[j] = 1.0
                etrans[i] = 1.0
            

    while True:
        has_new = False
        
        for i in range(elen):
            for j in range(flen+1):
                if isfinite(alignment[i][j]):
                    ## look at surrounding points
                    for k in range(num_neighbors):
                        left  = neighbors[k][0]
                        right = neighbors[k][1]

                        ## neighbors are not meaningful 
                        if (i+left < 0) or (j +right < 0) or (i+left >= elen) or (j + right >= flen+1): continue
                        if (isinf(etrans[i+left]) or isinf(ftrans[j+right])) and isfinite(union_a[i+left][j+right]):
                            alignment[i+left][j+right] = 1.0
                            etrans[i+left] = 1.0
                            ftrans[j+right] = 1.0
                            has_new = True

        if not has_new:
            break 
                                
    ## final(a) 
    for i in range(elen):
      for j in range(flen+1):
        if (isinf(etrans[i]) or isinf(ftrans[j])) and isfinite(union_a[i][j]):
          alignment[i][j] = 1.0
          etrans[i] = 1.0
          ftrans[j] = 1.0



cdef inline void _grow_diag_final_and(int[:] ftoe,int[:] etof,double[:,:] alignment,
                                  double[:,:] union_a,int flen,int elen):
    """starting point for starting grow heuristics"""
    cdef int etranslation
    cdef int i,j,k
    cdef list neighbors = [[-1,0],[0,-1],[1,0],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
    cdef int num_neighbors = len(neighbors)
    cdef int left,right
    cdef double[:] etrans = np.ndarray((elen,),dtype='d')
    cdef double[:] ftrans = np.ndarray((flen+1,),dtype='d')
    cdef bint has_new

    etrans[:] = np.inf
    ftrans[:] = np.inf

    ## fill alignment matrix with e->f
    _swap(etof,alignment,union_a,flen)

    ## intersect the main alignment 
    for i in range(elen):
        etranslation = ftoe[i]
        for j in range(flen+1):
            ## change if not consistent with f->e
            if isfinite(alignment[i][j]) and j != etranslation:
                alignment[i][j] = np.inf
            elif j == etranslation:
                union_a[i][j] = 1.0

            ## make has alignment set
            if isfinite(alignment[i][j]):
                ftrans[j] = 1.0
                etrans[i] = 1.0
            

    while True:
        has_new = False
        
        for i in range(elen):
            for j in range(flen+1):
                if isfinite(alignment[i][j]):
                    ## look at surrounding points
                    for k in range(num_neighbors):
                        left  = neighbors[k][0]
                        right = neighbors[k][1]

                        ## neighbors are not meaningful 
                        if (i+left < 0) or (j +right < 0) or (i+left >= elen) or (j + right >= flen+1): continue
                        if (isinf(etrans[i+left]) or isinf(ftrans[j+right])) and isfinite(union_a[i+left][j+right]):
                            alignment[i+left][j+right] = 1.0
                            etrans[i+left] = 1.0
                            ftrans[j+right] = 1.0
                            has_new = True

        if not has_new:
            break 
                                
    ## final(a) 
    for i in range(elen):
      for j in range(flen+1):
        if (isinf(etrans[i]) and isinf(ftrans[j])) and isfinite(union_a[i][j]):
          alignment[i][j] = 1.0
          etrans[i] = 1.0
          ftrans[j] = 1.0
            
