#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementations of the ``pivot-based`` bi-lingual paraphrase model using
Bannard and Callison-burch method

The method uses the suffix-array datastructure implemented in Zubr.Datastructures

"""
import sys
import os
import logging
import pickle
import numpy as np
cimport numpy as np
from zubr.util import ConfigAttrs
from zubr.Datastructures cimport ParallelCorpusSuffixArray,Indexer
from zubr.Alg cimport binary_insert_sort
from zubr.SymmetricAlignment cimport from_giza,SymmetricWordModel,SymAlign,Phrases
from collections import defaultdict
from cython cimport boundscheck,wraparound,cdivision
from zubr.util.paraphrase_util import build_suffix
from zubr.util.aligner_util import load_aligner_data,get_tree_data,load_glue_grammar
from zubr.ZubrClass cimport ZubrSerializable
from zubr.pysrilm import srilm


cdef class ParaphraseBase(ZubrSerializable):

    """A base class for building paraphrase models """

    cpdef list find_paraphrases(self,input_query,top_k=10):
        """paraphrase a given input

        :param input_query: the input to paraphrase
        :param top_k: the number of paraphrases to return 
        """
        raise NotImplementedError()

    @classmethod
    def build_paraphraser(cls):
        """Build a paraphrase model instance 

        :returns: paraphrase instance 
        """
        raise NotImplementedError()

cdef class MultilingualParaphraser(ParaphraseBase):

    property corpora_set:

        """return a string representation of corpus collection"""

        def __get__(self):
            raise NotImplementedError()

cdef class PivotParaphraser(MultilingualParaphraser):

    """A pivot paraphrase model built from a set of bilingual parallel data"""

    def __init__(self,corpora,lowercase=True,sample=200):
        """

        :param corpora: list of corpora in paraphrase model
        :type corpora: list (of suffix arrays)
        """
        self.corpora = corpora
        self.lowercase = lowercase
        self.sample_size = sample
        
    @classmethod
    def build_paraphraser(cls,datasets,lower=True,sample=250):
        """build a paraphrase model from a configuration

        :param datasets: the main datasets
        """
        return cls(datasets,lowercase=lower,sample=sample)


    cpdef list find_paraphrases(self,input_query,max=50,max_phrase=8):
        """find paraphrases given an english input


        :param input_query: the (english) query to paraphrase
        :type input_query: basestring
        :param max: the maximum number of paraphrases to return
        :type max: int
        :returns: list of candidates with scores
        :rtype: list of tuples
        """
        query = input_query.strip()
        if self.lowercase: query = query.lower()
        return generate_paraphrases(query,self.corpora,self.sample_size,max,max_phrase)
    
    def merge_models(self,other_model):
        """merge another paraphrase into this current instance

        :param other_model: another paraphrase model
        :type other_model:  PivotParaphraser
        :rtype: None                
        """        
        self.corpora += other_model.corpora
        
    property corpora_set:
        """return a string representation of corpus collection"""

        def __get__(self):
            ## pickle implementation
            return '\n'.join(["corpus %d: %s, %d sentences" % (k,i.name,i.size) \
                              for k,i in enumerate(self.corpora)])

    property sample_s:
    
        """sample size is the number of translations to paraphrase from"""

        def __get__(self):
            """returns the current sameple size"""
            return self.sample_size

        def __set__(self,int i):
            """resets the sample size"""
            self.sample_size = i

    def __reduce__(self):
        ## pickle implementation
        return PivotParaphraser,(self.corpora,self.lowercase,
                                 self.sample_size)

    ## backup protocol

    def backup(self,wdir):
        """Back up the paraphrase model to file 

        :param wdir: the working directory 
        """
        pdir = os.path.join(wdir,"paraphrases")
        info = os.path.join(pdir,"info.txt")
        if os.path.isdir(pdir):
            self.logger.info('Already backed up, skipping...')
            return

        ## make the backup directory 
        os.mkdir(pdir)
        
        for k,item in enumerate(self.corpora):
            item.backup(pdir,str(k))

        ## backup number
        with open(info,'w') as my_info:
            print >>my_info,len(self.corpora)
            print >>my_info,self.lowercase
            print >>my_info,self.sample_size
            
    @classmethod
    def load_backup(cls,config):
        """Load the backup

        :param config: the main configuration 
        """
        pdir = os.path.join(config.dir,"paraphrases")
        pinfo= os.path.join(pdir,'info.txt')
        orig_dir = config.dir
        corpora = []
        size = lowercase = sample_size = None
        
        ## the size or number of corpora 
        with open(pinfo) as my_info:
            fsplit = my_info.readlines()
            size = int(fsplit[0])
            lowercase = bool(fsplit[1])
            ssize = int(fsplit[2])

        config.dir = pdir
        for i in range(size):
            c = ParallelCorpusSuffixArray.load_backup(config,str(i))
            corpora.append(c)

        return cls(corpora,lowercase=lowercase,sample=ssize)

# paraphrase object to store paraphrases and scores

cdef class CounterBase:

    cdef unicode pop(self):
        """Return the next item from agenda

        :returns: the next item
        :rtype: unicode
        """
        cdef set agenda = self.agenda
        cdef unicode next_item

        try: 
            next_item = agenda.pop()
        except KeyError:
            return None
        return next_item

cdef class PhraseCounter:

    def __init__(self,num_corpora):
        """Creates a phrase counter instance

        :param num_corpora: the number of corpora to look at
        """
        self.corpus_occ = {i:0 for i in range(num_corpora)}
        self.phrase_occ = {i:{} for i in range(num_corpora)}
        self.agenda = set()

    cdef void add_phrase(self,int cnum,unicode phrase):
        """Add a phrase to the counts for a corpus

        :param cnum: the corpus number
        :param phrase: the phrase to add
        """
        cdef dict corpus_c = self.corpus_occ
        cdef dict cphrases = self.phrase_occ
        cdef set agenda = self.agenda

        #agenda.add(phrase)
        corpus_c[cnum] += 1
        
        if phrase not in cphrases[cnum]:
            cphrases[cnum][phrase] = 1.0 
        else:
            cphrases[cnum][phrase] += 1.0

    cdef void add_agenda(self,unicode phrase):
        """Add a translated phrase to the agenda

        :param phrase: the translated foreign phrase
        """
        cdef set agenda = self.agenda
        agenda.add(phrase)

cdef class ForeignPhraseCounter(CounterBase):

    def __init__(self,num_corpora):
        """"""
        self.corpus_occ = {i:{} for i in range(num_corpora)}
        self.phrase_occ = {i:{} for i in range(num_corpora)}
        self.agenda = set()

    cdef void add_foreign(self,int cnum, unicode candidate, unicode trigger):
        """add a foreign phrase

        :param cnum: the corpus number
        :param candidate: the e phrase to add
        :param trigger: the foreign trigger 
        """
        cdef dict corpus_c = self.corpus_occ
        cdef set agenda = self.agenda
        cdef dict cphrases = self.phrase_occ

        agenda.add(candidate)

        ## add trigger
        if trigger not in corpus_c[cnum]:
            corpus_c[cnum][trigger] = 1.0
        else:
            corpus_c[cnum][trigger] += 1.0

        ## add trigger + candidate
        if trigger not in cphrases[cnum]:
            cphrases[cnum][trigger] = {}

        if candidate not in cphrases[cnum][trigger]:
            cphrases[cnum][trigger][candidate] = 1.0
        else:
            cphrases[cnum][trigger][candidate] += 1.0
            
## overall paraphrase manager

cdef class ParaphraseManager:

    """A paraphrase manager class for computing paraphrase probabilities, counting
    translations, ..."""

    def __init__(self,num_corpora):
        """creates an e and f counter 

        :param num_corpora: the number of corpora used for paraphrases
        """
        self.ecounter = PhraseCounter(num_corpora)
        self.fcounter = ForeignPhraseCounter(num_corpora)
        self.num_corpora = num_corpora

    cdef list compute_probabilities(self,int maxsize):
        """assign a probability to the each candidate paraphrase

        :param maxsize: the maximum number of paraphrases to return
        """
        return rank_paraphrases(self.ecounter,self.fcounter,self.num_corpora,maxsize)


###

cdef class CorpusParaphraser(ZubrSerializable):
    """A class for paraphrasing a corpus"""

    def __init__(self,model,paraphraser,lm=None):
        """Initializes a CorpusParaphrser instance 

        :param model: the translation model 
        :param paraphraser: the paraphraser container 
        """
        self.model       = model
        self.paraphraser = paraphraser
        self.lm          = lm

    def paraphrase_corpus(self,config):
        """Main python method for paraphrasing a corpus

        :param config: the main configuration 
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls,config):
        """Creates a GrammarParaphraser instance from config

        :param config: the main configuration 
        """
        model = SymmetricWordModel.load_backup(config)
        paraphrases = PivotParaphraser.load_backup(config)

        return cls(model,paraphrases,None)

cdef class PhraseParaphraser(CorpusParaphraser):
    """This paraphraser simply paraphraser entries from the raw phrase table, without 
    any additional information. 
    """
    pass 

cdef class GrammarParaphraser(CorpusParaphraser):
    """This is a type of corpus paraphraser that uses a glue grammar to specify 
    the type of phrases that should be paraphrased. 

    It find phrases is a manner very similar to how our hiero extraction procedure
    works, and repeats a lot of this code (this should somehow be one implementation, 
    at the moment there is too much redundancy!!)

    What is assumed, in addition to the training data, is a file called <data_name>.tree, 
    which specifies the abstract trees sequence for each training representation, as well as 
    a file called grammar.txt, which specifies how to ``glue`` together these abstract tree
    sequences. 

    For example, let's assume that we have the following pair of (descriptions,reps):

    ( "my function with arg1", internal function1 v_arg1 )

    And let's assume that the rep has the following sequence, sepcified in <data_name>.tree: 

    internal function1 v_arg1 -> 0 0 1 

    Okay, so the number don't mean much, but in our glue grammar, we might have: 

    function_name -> 0 
    function_name -> 0
    argument_name -> 1
    fun_arg -> function_name argument_name 
    
    This spepcifies what these sequence mean, but also specify constraints on what phrases to 
    paraphrase. For example, let's assume that there is a single phrase that spans 0 -> 1, we will 
    not immediately extract that phrase because we must find a function_name and argument_name first before
    joining them. This is avoid extracting unreasonably long phrases. 

    """
    def paraphrase_corpus(self,config):
        """Main python method for paraphrasing a corpus

        :param config: the main configuration 
        """

        ## the main data 
        data = load_aligner_data(config)
        f = data[0]; e = data[1]
        dsize = f.shape[0]

        ## the tree data
        tree_pos = get_tree_data(config,tset='train')

        ## log data infro
        self.logger.info('datasize=%d, tree size=%d' % (e.shape[0],tree_pos.shape[0]))

        

### c methods

@cdivision(True)
cdef list rank_paraphrases(PhraseCounter ec,ForeignPhraseCounter fc,int cnum,int maxs):
    cdef int i,j
    cdef dict e_f_occ = ec.phrase_occ
    cdef dict e_c_occ = ec.corpus_occ
    cdef dict f_c_occ = fc.corpus_occ
    cdef dict f_e_occ = fc.phrase_occ
    cdef int len_can = len(fc.agenda)

    ## paraphrase scores
    cdef double[:] can_scores = np.zeros((len_can,),dtype='d')
    cdef list candidates = list(fc.agenda)
    cdef double e_occ,f_occ,cann_score,fcount
    cdef double f_e_prob,e_f_prob,can_occ

    ## rank list
    cdef int[:] sort = np.ndarray((len_can,),dtype=np.int32)
        
    ## candidate 
    cdef unicode candidate,fphrase
    cdef float c_nonzero = 0.0

    ## can translations
    cdef dict candidate_trans = {}

    sort[:] = -1

    ## go through each corpus 
    for i in range(cnum):
        e_occ = e_c_occ[i]
        if e_occ == 0.0: continue
        c_nonzero += 1.0

        for j in range(len_can):
            candidate = candidates[j]
            cann_score = 0.0

            if candidate not in candidate_trans:
                candidate_trans[candidate] = set()

            for (fphrase,fcount) in e_f_occ[i].items():

                ## some translations only translate to original phrase, therefore have 0 occurrence
                f_occ = f_c_occ[i].get(fphrase,0.0)
                if f_occ == 0: continue
                
                ## p(f | e ) = e_f_occ[i][fphrase]/e_occ
                
                ## check if candidate is here
                can_occ = f_e_occ[i][fphrase].get(candidate,0.0)
                if can_occ > 0.0:

                    ## compute both probabilities
                    # p(e' | f) = cann_occ/f_occ
                    # p(e' | e) =  (cann_occ/f_occ)*(e_f_occ[i][fphrase]/e_occ)
                    cann_score += (can_occ/f_occ)*(e_f_occ[i][fphrase]/e_occ)
                    candidate_trans[candidate].add(fphrase)
                
            can_scores[j] += cann_score

    ## sort the resulting candidate list (binary sort)
    for i in range(len_can):
        #cann_score = can_scores[i]/float(cnum)
        cann_score = can_scores[i]/c_nonzero
        can_scores[i] = cann_score
        binary_insert_sort(i,cann_score,can_scores,sort)

    if maxs > len_can:
        maxs = len_can
    return [(candidates[sort[i]],candidate_trans[candidates[sort[i]]],can_scores[sort[i]]) for i in range(maxs)]

@cdivision(True)
@boundscheck(False)                       
cdef list generate_paraphrases(p_input,list corpus_set,int sample_size,int max_size,int max_phrase):
    """generate a set of paraphrases given some input

    :param p_input: the paraphrase input
    :param corpus_set: set of corpora to query
    :param sample_size: the number of translations to consider 
    :param max_size: the maximum number of paraphrases to return
    """
    cdef ParallelCorpusSuffixArray corpus
    cdef Indexer eindex,findex
    cdef int i,j,num_corpora = len(corpus_set)
    cdef int foreign_end,foreign_start
    cdef int k,estart,eend,fstart,fend
    cdef double corpus_occ
    cdef int qlen = len(p_input.split())
    cdef int[:,:] estarts,fstarts,starts
    cdef int[:] esort_list,fsort_list
    cdef int sentence_id,espos,esort_id,fsort_id,eespos
    cdef int fspos,fespos
    cdef np.ndarray fraw,eraw
    cdef list alignments
    cdef int[:,:] efphrases,fephrases
    cdef np.ndarray fstring,estring
    cdef unicode ftrans,etrans
    cdef int minj,maxj

    ## paraphrase counters
    cdef ParaphraseManager manager = ParaphraseManager(num_corpora) 
    cdef PhraseCounter ecounter = manager.ecounter
    cdef ForeignPhraseCounter fcounter = manager.fcounter 

    for i in range(num_corpora):

        ## corpus instance
        corpus = corpus_set[i]
        
        ## alignment list
        alignments = corpus.alignment
                
        ## raw corpus
        fraw = corpus.foreign_data
        eraw = corpus.english_data
        esort_list = corpus.english_sorted
        fsort_list = corpus.foreign_sorted

        ## span of query in english set
        eindex = corpus.query(p_input)
        estart = eindex.start
        eend = eindex.end

        ## general datstructures
        estarts = corpus.esentences
        fstarts = corpus.fsentences
        starts  = corpus.start_pos
        
        ## no ocurrence in this corpus
        if estart < 0 or eend < 0:
            continue

        ## check that number of translations is less than sample_size

        if (eend - estart) > max_size:
            eend = estart + max_size

        ## each foreign phrase occurrence in corpus 
        for k in range(estart,eend+1):

            ## english prefix identity or position in corpuss 
            esort_id = esort_list[k]
            ## sentence number 
            sentence_id = estarts[esort_id][0]
            ## position in the target sentence of first word
            espos = estarts[esort_id][1]
            ## end position
            eespos = estarts[esort_id][1]+(qlen-1)

            ## check if span is in the same sentence..?
            if qlen > 1 and sentence_id != estarts[esort_id+(qlen-1)][0]:
               continue

            fsort_id = starts[sentence_id][1]
            ## extract phrases for spot (espos,eespos) 
            efphrases = from_giza(alignments[sentence_id],starts[sentence_id][2],
                                 starts[sentence_id][3],espos,eespos)

            ## the corresponding foreign sentence
            fstring = fraw[fsort_id:fsort_id+(starts[sentence_id][3])]

            #minj = fstring.shape[0]
            #maxj = -1

            ## back to choosing all
            for j in range(<int>efphrases.shape[0]):

                if (efphrases[j][3] - efphrases[j][2]-1) >= max_phrase:
                    continue  
                                
                ftrans = ' '.join(fstring[efphrases[j][2]-1:efphrases[j][3]])
                ecounter.add_agenda(ftrans)

                #ftrans = ' '.join(fstring[])

            
            ## find the maximal phrase (do not choose all phrases) 

            
            # for j in range(<int>efphrases.shape[0]):
            #     if efphrases[j][2]-1 < minj:
            #         minj = efphrases[j][2]-1
            #     if efphrases[j][3] > maxj:
            #         maxj = efphrases[j][3]

            # if maxj != -1:
            #     ftrans = ' '.join(fstring[minj:maxj])
            #     ecounter.add_agenda(ftrans)
                #ecounter.add_phrase(i,ftrans)

        ## go through phrase and find their translations

        while True:

            ftrans = ecounter.pop()
            if not ftrans: break

            qlen = len(ftrans.split())
            ## find this in the foreign side of the corpus 
            findex = corpus.query_foreign(ftrans)
            foreign_start = findex.start
            foreign_end = findex.end
            
            ## shouldn't be the case,
            if foreign_end < 0 or foreign_start < 0:
                continue

            ## limit the number of translations to consider
            
            if (foreign_end - foreign_start) > max_size:
                foreign_end = foreign_start + max_size 

            ## go through each foreign instance 
            for k in range(foreign_start,foreign_end+1):

                ## foreign sort id
                
                fsort_id = fsort_list[k]
                ## sentence_id
                sentence_id = fstarts[fsort_id][0]

                ## position in the foreign sentence of first word
                fspos = fstarts[fsort_id][1]
                fespos = fstarts[fsort_id][1]+(qlen-1)

                if qlen > 1 and sentence_id != fstarts[fsort_id+(qlen-1)][0]:
                    continue

                esort_id = starts[sentence_id][0]
                
                fephrases = from_giza(alignments[sentence_id],
                                      starts[sentence_id][3],
                                      starts[sentence_id][2],
                                      fspos,
                                      fespos,reverse=True)

                estring = eraw[esort_id:esort_id+(starts[sentence_id][2])]

                ## pick the maximum phrase 
                #minj = estring.shape[0]
                #maxj = -1

                ## go through the phrases 
                for j in range(<int>fephrases.shape[0]):

                    ## put restriction on phrase size
                    if (fephrases[j][3]-1 - fephrases[j][2]) >= max_phrase:
                        continue
                    
                    ## go back to considering all 
                    etrans = ' '.join(estring[fephrases[j][2]-1:fephrases[j][3]])

                    if etrans != p_input:
                        fcounter.add_foreign(i,etrans,ftrans)
                        ecounter.add_phrase(i,ftrans)

                ## picking the largest ones
                  
                #     if fephrases[j][2]-1 < minj:
                #         minj = fephrases[j][2]-1
                #     if fephrases[j][3] > maxj:
                #         maxj = fephrases[j][3]

                # if maxj != -1: 
                #     etrans = ' '.join(estring[fephrases[j][2]-1:fephrases[j][3]])
                #     etrans = ' '.join(estring[minj:maxj])
                #     if etrans != p_input:
                #         fcounter.add_foreign(i,etrans,ftrans)
                #         ecounter.add_phrase(i,ftrans)

    ## compute paraphrase scores using manager
    return manager.compute_probabilities(max_size)
        
                    
#### CLI methods

PARAS = {
    "grammar" : GrammarParaphraser,
    "phrase"  : PhraseParaphraser,
}

def Paraphraser(ptype):
    """Factory method for returning a paraphrase model 

    :param ptype: the particular paraphrase model 
    :raises: ValueError 
    """
    pmodel = PARAS.get(ptype,None)
    if not pmodel:
        raise ValueError('Unknown type of paraphraser: %s' % str(ptype))
    return pmodel


def params():
    """main parameters for running and setting up a paraphrase model

    :rtype: tuple
    :returns: description of option types with name, list of options 
    """
    options = [
        ("--save_model","save_model",True,'bool',
         "save the paraphrase model [default='True']","Paraphraser"),
        ("--sample_size","sample_size",250,'int',
         "number of translations to consider [default='200']","Paraphraser"),
        ("--parallel","parallel",1,'int',
         "number of concurrent sorts (linux only) [default='5']","Paraphraser"),
        ("--preprocess","preprocess",False,'bool',
         "preprocess the raw paraphrase data [default=False]","Paraphraser"),
        ("--paraphrase_corpus","paraphrase_corpus",False,'bool',
         "Paraphrase a given corpus [default=False]","Paraphraser"),
        ("--pmodel","pmodel",'grammar','str',
         "The type of paraphrase model to use [default='grammar']","Paraphraser"),
        ("--paraphrase_data","paraphrase_data",'','str',
         "Parallel data to build paraphrase model [default='']","Paraphraser"),         
    ]

    paraphrase_group = {'Paraphraser':'Paraphrase settings and defaults'}
    return (paraphrase_group,options)

def argparser(extras=None):
    """return an paraphrase parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default paraphrase parser
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr paraphrase [options]"""
    d,options = params()
    if extras:
        options += extras
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser


def main(argv):

    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        from zubr.SymmetricAligner import params as sparams
        parser = argparser(sparams()[-1])
        config = parser.parse_args(argv)
        logging.basicConfig(level=logging.DEBUG)

    if config.paraphrase_corpus:
        ## build a corpus paraphraser model 
        pclass = Paraphraser(config.pmodel)
        paraphraser = pclass.from_config(config)

        ## extract
        paraphraser.paraphrase_corpus(config)
        
    else:
        corpora = build_suffix(config)
        ## replace with a factory method 
        paraphrase_model = PivotParaphraser.build_paraphraser(corpora,lower=config.lower,
                                                            sample=config.sample_size)

    try:
        paraphrase_model.backup(config.dir)
    except:
        pass
