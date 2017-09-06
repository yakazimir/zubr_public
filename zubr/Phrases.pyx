#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Classes for representing different types of phrases

"""
import re
import os
import sys
import traceback
import logging
import gzip
import codecs
import time
import pickle 
from collections import defaultdict
from string import punctuation
from zubr.util.stops import stops
from zubr.Features cimport FeatureMap
from zubr.ZubrClass cimport ZubrSerializable
from zubr.Datastructures cimport Sparse2dArray
from zubr.util.phrase_util import *
from zubr.Dataset cimport Dataset,RankPair
import numpy as np
cimport numpy as np
from cython cimport wraparound,boundscheck,cdivision

cdef bint is_stop(str lang,unicode word):
    """Check if a word is a stop word

    :param lang: the target language
    :param word: the word to check
    """
    cdef dict stop_list = stops
    cdef set lang_list = stops.get(lang,set([]))
    if not lang_list: return False
    return <bint>(word in lang_list)

cdef class TranslationSide:
    
    """Absttact class for representating translation sides"""
    
    def __str__(self):
        ## must have string implementation 
        raise NotImplementedError

    def __reduce__(self):
        ## must have pickle implementation
        raise NotImplementedError

## side of a raw phrase or word pair

cdef class PhraseSide(TranslationSide):

    def __init__(self,side):
        self.string = side.strip()

    def __str__(self):
        return <unicode>self.string

    def __reduce__(self):
        return PhraseSide,(self.string,)


########### SIDE of hierarchical phrase rule 
    
cdef class HieroSide:

    """Object for dealing with one side of synchronous hiero rule"""

    def __init__(self,rule):
        """ 
        
        :param rule: the input rule
        :type rule: unicode
        """
        self.rule   = rule
        self.nts    = [(i,k) for k,i in enumerate(rule.split()) if re.search(r'\[.+\]',i)]
        self.string = re.sub(r'\s+',' ',re.sub(r'\[.*?\]','',rule).strip())

    property left_context:
        """the left context (if exists), or words to left of terminal symbols
         
        :note: returns the empty string if no nts exists
        """

        def __get__(self):
            """get the left context

            :rtype: unicode
            """
            cdef list nts = self.nts
            cdef unicode rule = self.rule,w
            if not nts: return u''

            return u' '.join([w for k,w in enumerate(rule.split()) if k < nts[0][1]])
            
    property right_context:
        """the right context (if exists), or words to right of terminal symbols

        :note: returns the empty string if no nts exists
        """

        def __get__(self):
            """get the right context

            :rtype: unicode
            """
            cdef list nts = self.nts
            cdef unicode rule = self.rule,w
            if not nts: return u''

            return u' '.join([w for k,w in enumerate(rule.split()) if k > nts[-1][1]])

    property middle_context:
        """the middle context (if exists), or words in between terminal symbols

        :note: returns the empty string if no nts exists
        """

        def __get__(self):
            """get the middle context

            :rtype: unicode
            """
            cdef list nts = self.nts
            cdef unicode rule = self.rule,w
            if not nts: return u''

            return u' '.join([w for k,w in enumerate(rule.split()) \
                                  if k in range(nts[0][1]+1,nts[-1][1])])

    cpdef int context_size(self):
        """Returns the total size of the surrounding word contexts

        :rtype int
        """
        cdef unicode l = self.left_context
        cdef unicode r = self.right_context
        cdef unicode m = self.middle_context

        return len(l.split())+len(r.split())+len(m.split())
        
    def __str__(self):
        # string implementation
        return <unicode>self.rule

    def __reduce__(self):
        ## pickle implementation 
        return HieroSide,(self.rule,)


    
cdef class TranslationPair:

    """Abstract class for pair of strings"""

    @classmethod 
    def from_tuple(cls,input_tuple):
        """Loads a translation pair from a tuple 

        :rtype: TranslationPair
        """
        raise NotImplementedError

    @classmethod
    def from_str(cls,str_input):
        """Loads a translation pair from string 

        
        :rtype: TranslationPair
        """
        raise NotImplementedError

    @classmethod
    def from_pair(cls,s1,s2):
        """Loads a translation pair from two unicode strings

        :param s1: the first input string 
        :type s1: unicode 
        :param s2: the second input string
        :type s2: unicode
        :rtype: TranslationPair
        """
        raise NotImplementedError

    def tuple_rep(self):
        """Returns a tuple representation of the translation pair

        :rtype: tuple
        """
        raise NotImplementedError

    cpdef bint sides_match(self):
        """Deteremines if translation pairs match


        :rtype: bool
        """
        raise NotImplementedError

    cpdef bint econtainsf(self):
        """Checks if the e string contains f string

        :rtype: bool
        """
        raise NotImplementedError

    cpdef bint fcontainse(self):
        """Checks if the f string contains e string

        :rtype: bool
        """
        raise NotImplementedError


    #cpdef bint word_overlap(self):
    cpdef int word_overlap(self):
        """Check if words overlap

        :rtype bool
        """
        raise NotImplementedError

    cdef bint is_stop_word(self):
        """Checks if the english word is a stop word

        :rtype: bool
        """
        raise NotImplementedError

    def __richcmp__(self,other,opt):
        ## implementation of comparisons
        raise NotImplementedError

    def __reduce__(self):
        ## must have pickle implementation
        raise NotImplementedError


cdef class PhrasePair(TranslationPair):

    def __init__(self,english,foreign,lang='en',identifier=-1,eid=-1,fid=-1):
        """ 

        :param english: the english side of phrase 
        :param foreign: the foreign side of phrase 
        :param lang: the language
        """
        self.english = english
        self.foreign = foreign
        self.lang    = lang
        self.num     = identifier
        self.eid     = eid
        self.fid     = fid

    @classmethod
    def from_str(cls,string_input):
        """Build a PhrasePair instance from a tab delimited unicode string

        :param string_input: 
        :type string_input: unicode (should update to do conversion)
        """
        tsplit = string_input.split('\t')
        english = tsplit[0].strip()
        foreign = tsplit[1].strip()

        return cls(PhraseSide(english),PhraseSide(foreign))

    @classmethod
    def from_pair(cls,s1,s2):
        """Build a PhrasePair instance from two unicode string 

        :param s1: the left side string
        :type s1: unicode
        :param s2: the right side string
        :type s2: unicode 
        """
        return cls(PhraseSide(s1.strip()),PhraseSide(s2.strip()))
        
    @classmethod
    def from_tuple(cls,tuple_input):
        english = ' '.join(tuple_input[0])
        foreign = ' '.join(tuple_input[1])
        
        return cls(PhraseSide(english.strip()),PhraseSide(foreign.strip()))

    def tuple_rep(self):
        """Create a tuple representation of a given rule

        :rtype: tuple 
        """
        # return (tuple(self.english.string.split()),
        #         tuple(self.foreign.string.split()))
        return (self.english.string,self.foreign.string)

    cpdef bint sides_match(self):
        """check that side of pair match

        :rtype: bint
        """
        cdef unicode e = self.english.string
        cdef unicode f = self.foreign.string
        if f == u'' or e == u'': return False
        
        return <bint>(e == f)

    cpdef bint econtainsf(self):
        """Check if the english string contains the foreign string

        :rtype: bint 
        """
        cdef unicode e = self.english.string
        cdef unicode f = self.foreign.string
        if f == u'' or e == u'': return False

        return <bint>(e.__contains__(f) and <int>(len(f)) > 2)

    cpdef bint fcontainse(self):
        """Check if the english string contains the foreign string

        :rtype: bint 
        """
        cdef unicode e = self.english.string
        cdef unicode f = self.foreign.string
        if f == u'' or e == u'': return False
        
        return <bint>(f.__contains__(e) and <int>(len(e)) > 2)

    cpdef int word_overlap(self):
        """Check if words in raw pair overlap

        :rtype: bint 
        """
        cdef unicode e = self.english.string
        cdef unicode f = self.foreign.string

        return len(set(e.split()) & set(f.split()))
        #return <bint>(set(e.split()) & set(f.split()))

    cdef bint is_stop_word(self):
        """Checks if the english side of the pair is a stop word

        -- Note: if not stoplist is available, returns False

        :rtype: bint 
        """
        cdef str lang = self.lang
        cdef unicode e = self.english.string
        return is_stop(lang,e)

    ### magic methods

    def __richcmp__(self,other,opt):

        if opt == 2:
            return (self.english.string == other.english.string) and\
              (self.foreign.string == other.foreign.string)

        raise TypeError('comparison for Rule not supported')

    def __reduce__(self):
        return PhrasePair,(self.english,self.foreign)

    def __str__(self):
        return u"%s\t%s" % (self.english,self.foreign)

    ## property

    property  is_known:
        """Return information about the phrase identifier """
        
        def __get__(self):
            """Returns whether the prhase is known or not

            :rtype: bool
            """
            cdef int identifier = self.num
            return (identifier != -1)
        
    
    
cdef class HieroRule(TranslationPair):

    """A representation of hierarchical phrase rules"""
    
    def __init__(self,lhs,erhs,frhs,freq=1,rule_number=-1,lang='en',eid=-1,fid=-1):
        """ 

        :param lhs: left hand side of synchronous rule
        :param erhs: the english right hand side
        :param frhs: the foreign right hand side 
        :param freq: frequency of the rule 
        :param rule_number: identifier in the symbol table (if avail)
        """
        self.lhs  = lhs
        self.erhs = erhs
        self.frhs = frhs
        self.freq = freq
        self.rule_number = rule_number
        self.lang = lang

        ## side ids
        self.eid = eid
        self.fid = fid 

    @classmethod
    def from_str(cls,str_input,rule_num=-1):
        """Parse a representation from a string 
        
        :param str_input: the rule in string form
        :type str_input: unicode 
        :rtype: HieroRule
        """
        lhs,rhs,freq = str_input.split('\t')
        english,foreign = [r.strip() for r in rhs.split(' ||| ')]
        freq_int = int(freq.split('=')[-1])
        rhs_e = HieroSide(english)
        rhs_f = HieroSide(foreign)

        return cls(lhs,rhs_e,rhs_f,freq=freq_int,rule_number=rule_num)

    @classmethod
    def from_tuple(cls,input_tuple):
        """Construct a class from a tuple representation
        
        :param input_tuple: the input rule in tuple form 
        :rtype: HieroRule 
        """
        lhs,side1,side2 = input_tuple
        rhs1 = HieroSide(' '.join(side1))
        rhs2 = HieroSide(' '.join(side2))
        
        return cls(lhs,rhs1,rhs2)

    def tuple_rep(self):
        """Create a tuple representation of a given rule

        :rtype: tuple 
        """
        return (self.lhs,self.erhs.rule,self.frhs.rule)

    cpdef bint sides_match(self):
        """Check if the strings sides of synchronous rule matches

        :rtype: bool
        """
        cdef unicode e = self.erhs.string, f = self.frhs.string
        if not e or not f: return False
        return <bint>(e == f)

    cpdef bint econtainsf(self):
        """Check is estring contains fstring
        
        :rtype: bool
        """
        cdef unicode e = self.erhs.string,f = self.frhs.string
        if not e or not f: return False
        return <bint>(e.__contains__(f) and <int>len(f) > 2)

    cpdef bint only_terminals(self):
        """Checks if rule only consists of 

        :rtype: bool
        """
        cdef unicode e = self.erhs.string
        cdef unicode f = self.frhs.string

        return <bint>(not e and not f)

    cpdef bint left_terminal_only(self):
        """Checks if left english side if a terminal only 

        :rtype: bool
        """
        cdef unicode e = self.erhs.string.strip()

        return <bint>(not e)

    cpdef bint right_terminal_only(self):
        """Checks if left foreign side if a terminal only 

        :rtype: bool
        """
        cdef unicode f = self.frhs.string.strip()

        return <bint>(not f)
        

    cdef bint is_stop_word(self):
        """Checks if the english side of the pair is a stop word

        -- Note: if not stoplist is available, returns False

        :rtype: bint 
        """
        cdef str lang = self.lang
        cdef unicode e = self.erhs.string
        return is_stop(lang,e)
    
    cpdef bint fcontainse(self):
        """Check is estring contains fstring
        
        :rtype: bool
        """
        cdef unicode e = self.erhs.string,f = self.frhs.string
        if not e or not f: return False
        return <bint>(f.__contains__(e) and <int>len(e) > 2)

    cpdef bint has_reordering(self):
        """Determines if non-terminal rules are re-ordered

        Note: assumes that each rule has _x number starting from 1,
        and that rules are binary 
        
        :rtype: bool
        """
        cdef list rhs = self.frhs.nts
        if not rhs: return False
        return <bint>(u'_' not in rhs[-1][0] or u'_1' in rhs[-1][0])

    cpdef bint left_contexts_match(self):
        """Check if the left contexts match"""
        cdef unicode eleft = self.erhs.left_context
        cdef unicode fleft = self.frhs.left_context

        return <bint>(eleft and fleft and eleft == fleft)
    
    cpdef bint right_contexts_match(self):
        """Check if the left contexts match"""
        cdef unicode eright = self.erhs.right_context
        cdef unicode fright = self.frhs.right_context

        return <bint>(eright and fright and eright == fright)

    #cpdef bint word_overlap(self):
    cpdef int word_overlap(self):
        """determines if word-overlap occurs between contexts"""
        cdef unicode e = self.erhs.string, f = self.frhs.string
        
        return len(set(e.split()) & set(f.split()))

    def __str__(self):
        # string implementation
        return u"%s\t%s ||| %s\tcount=%d" %\
          (self.lhs,str(self.erhs),str(self.frhs),self.freq)

    def __richcmp__(self,other,opt):

        if opt == 2:
            return (self.lhs == other.lhs) and (self.erhs.rule == other.erhs.rule) and\
              (self.frhs.rule == other.frhs.rule)

        raise TypeError('comparison for Rule not supported')

    def __reduce__(self):
        ## pickle implementation
        return HieroRule,(self.lhs,self.erhs,self.frhs,self.freq,self.rule_number)

    ### property

    property english_side:
        
        """returns the english side of the rule"""
        def __get__(self):
            return <unicode>self.lhs

    property foreign_side:

        """returns the foreign side of the rule"""

        def __get__(self):
            return <HieroSide>self.frhs

## c-level class method for building instances (the @classmetho python methods might be too slow)

cdef PhrasePair ConstructPair(unicode s1, unicode s2,str lang='en'):
    """Construct a phrase pair instance from two unicode strings

    :param s1: string 1
    :param s2: string 2
    :rtype: PhrasePair 
    """
    cdef PhraseSide side1, side2
    
    side1 = PhraseSide(s1.strip())
    side2 = PhraseSide(s2.strip())
    
    return <PhrasePair>PhrasePair(side1,side2,lang=lang)

cdef HieroRule ConstructHPair(unicode lhs, unicode s1,unicode s2,str lang='en'):
    """Constructs a hiero phrase from two unicode string and lhs id

    :param lhs: the left hand side of the hiero rule 
    :param s1: the english side of the rule
    :param s2: the foreign side of the rule
    """
    cdef HieroSide side1,side2
    
    lhs = lhs.strip()
    side1 = HieroSide(s1.strip())
    side2 = HieroSide(s2.strip())
    
    return <HieroRule>HieroRule(lhs,side1,side2,lang=lang)


## PHRASE TABLE SUFFIX ARRAY DATASTRUCTRE

## while there is a base suffix array implement in Datastructures.pyx,
## the design is pretty bad, and will need to be re-implemented, so
## I am building the phrase table here from scratch x

cdef unicode to_unicode(s):
    if isinstance(s,bytes):
        return (<bytes>s).decode('utf-8')
    return s

cdef class PhraseTableBase(ZubrSerializable):

    def __init__(self,sorted_list,raw_phrases,ids,elen,flen):
        """Create a SimplePhraseTable instance

        :param sorted_list: the alphabetically sorted list of phrases by index
        :param raw_phrases: the actual stored phrases 
        """
        self.slist    = sorted_list
        self.phrases  = raw_phrases
        self.ids      = ids
        self.elen     = elen
        self.flen     = flen
    
    def __iter__(self):
        ## iterate over phrase list 
        return iter(self.phrase_list)

    def print_rules(self,wout):
        """Print out the rules in this table 

        :param wout: the output path 
        """
        raise NotImplementedError
    
    cpdef int query(self,input1,input2,lhs=u""):
        """Query the phrase tables given a pair of phrases

        :param input1: the first input 
        :param input2: the second input (if needed) 
        """
        raise NotImplementedError

    @classmethod
    def create_empty(cls):
        """Creates an empty phrase table instance

        :returns: phrase instance
        """
        return cls(np.empty((0,),dtype=np.int32),
                       np.empty((0,),dtype=np.object),
                       np.empty((0,0),dtype=np.int32),
                       0,0)

    def backup(self,wdir):
        raise NotImplementedError

    @classmethod 
    def load_backup(cls,config):
        raise NotImplementedError

    property num_phrases:
        """Information about the number of phrases"""
        
        def __get__(self):
            """Return the number of phrases in table

            :rtype: int 
            """
            cdef np.ndarray phrase_list = self.phrases
            return <int>phrase_list.shape[0]

    property is_empty:
        """Determine if the list or container is empty"""
        
        def __get__(self):
            """Returns a boolean indiciating if container is empty or not 

            :rtype: bool
            """
            cdef np.ndarray phrase_list = self.phrases
            return (phrase_list.shape[0] == 0)
        
cdef class SimplePhraseTable(PhraseTableBase):
    """Base class and container for representating phrase tables using a suffix array"""

    cpdef int query(self,input1,input2,lhs=u""):
        """Query the phrase table for specific phrases

        :param input1: the left side of the phrase 
        :param input2: the right side of the phrase (if exists) 
        :returns: the index of the phrase in the table 
        """
        cdef int[:] alpha = self.slist
        cdef np.ndarray phrases = self.phrases
        cdef int pidentifier
        cdef unicode left,right,query
        cdef int pn = phrases.shape[0]

        left  = to_unicode(input1.strip())
        right = to_unicode(input2.strip())
        query = u"%s ||| %s" % (left,right)

        ## query if the the phrase is a valid one
        pidentifier = _query_table(query, phrases,alpha,pn)
        
        ## return back a phrase representation
        return pidentifier

    cdef PhrasePair create_pair(self,input1,input2,str lang='en'):
        """Create a phrase pair, returns also the identifier 

        :param input1: the left hand side of the phrase (typically e)
        :param input2: the right hand side of the phrase (typically f)
        """
        cdef int[:] alpha = self.slist
        cdef int[:,:] ids = self.ids 
        cdef np.ndarray phrases = self.phrases
        cdef unicode query,left,right
        cdef int pidentifier
        cdef int pn = phrases.shape[0]
        cdef PhraseSide side1,side2
        cdef int eid,fid

        left = to_unicode(input1.strip())
        right = to_unicode(input2.strip())
        side1 = PhraseSide(left)
        side2 = PhraseSide(right) 
        query = u"%s ||| %s" % (left,right)

        ## find the identifier 
        pidentifier = _query_table(query,phrases,alpha,pn)
        
        #eid,fid
        if pidentifier != -1: 
            eid = ids[pidentifier][0]
            fid = ids[pidentifier][1]
        else:
            eid = -1
            fid = -1

        return PhrasePair(side1,side2,lang=lang,identifier=pidentifier,eid=eid,fid=fid)

    @classmethod
    def from_config(cls,config):
        """Loads a phrase table item from configuration

        Note : Will look for a file: phrase_table.txt in your 
        working directory, and sort the items in that file. 

        :param config: the main configuration 
        :raises: ValueError
        """
        sorted_list,phrases,ids,elen,flen = preprocess_table(config)
        return cls(sorted_list,phrases,ids,elen,flen)

    def backup(self,wdir,name='simple_table'):
        """Backup the phrase components 

        :param wdir: the working directory 
        :rtype: None 
        """
        stime = time.time()
        simple_table = os.path.join(wdir,name)
        if os.path.isfile(simple_table+".npz"):
            self.logger.info('Already backed up, skipping...')
            return

        infoa = np.array([self.elen,self.flen],dtype=np.int32) 
        np.savez_compressed(simple_table,self.slist,self.phrases,self.ids,infoa)
        self.logger.info('Backed up in %s seconds' % (time.time()-stime))

    @classmethod
    def load_backup(cls,config,name='simple_table'):
        """Load a backup or instance from file 

        :param config: the global configuration 
        :returns: simple phrase table instance 
        """
        stime = time.time()
        simple_table = os.path.join(config.dir,name+".npz")
        archive = np.load(simple_table)
        slist   = archive["arr_0"]
        phrases = archive["arr_1"]
        ids     = archive["arr_2"]
        info    = archive["arr_3"]

        instance = cls(slist,phrases,ids,info[0],info[1])
        instance.logger.info('Loaded backup in %s seconds' % (time.time()-stime))
        return instance     

        
    def __reduce__(self):
        ## pickle implementation
        return SimplePhraseTable,(self.slist,self.phrases)

    
cdef class HieroPhraseTable(PhraseTableBase):
    """Phrase table for hierarchical phrase rules"""

    def __init__(self,sorted_list,raw_phrases,ids,elen,flen,glue):
        """Create a SimplePhraseTable instance

        :param sorted_list: the alphabetically sorted list of phrases by index
        :param raw_phrases: the actual stored phrases 
        """
        self.slist    = sorted_list
        self.phrases  = raw_phrases
        self.ids      = ids
        self.elen     = elen
        self.flen     = flen
        self.glue     = glue

        ## make a lhs map lookup 
        self.lhs_lookup = {i:k for k,i in enumerate(set(glue.values()))}
        
    @classmethod
    def from_config(cls,config):
        """Loads a phrase table item from configuration

        Note : Will look for a file: hiero_rules.txt in your 
        working directory, and sort the items in that file. 

        :param config: the main configuration 
        :raises: ValueError
        """
        (glue,(sorted_list,phrases,ids,elen,flen)) = preprocess_hiero(config)
        return cls(sorted_list,phrases,ids,elen,flen,glue)

    @classmethod
    def create_empty(cls):
        """Creates an empty phrase table instance

        :returns: phrase instance
        """
        return cls(np.empty((0,),dtype=np.int32),
                       np.empty((0,),dtype=np.object),
                       np.empty((0,0),dtype=np.int32),
                       0,0,{})

    cpdef int query(self,input1,input2,lhs=u""):
        """Query the phrase table for specific phrases

        :param input1: the left side of the phrase 
        :param input2: the right side of the phrase (if exists) 
        :returns: the index of the phrase in the table 
        """
        cdef int[:] alpha = self.slist
        cdef np.ndarray phrases = self.phrases
        cdef int pidentifier
        cdef unicode left,right,query
        cdef int pn = phrases.shape[0]

        left  = to_unicode(input1.strip())
        right = to_unicode(input2.strip())
        lhs = to_unicode(lhs.strip()) 
        query = u"%s ||| %s ||| %s" % (lhs,left,right)

        ## query if the the phrase is a valid one
        pidentifier = _query_table(query, phrases,alpha,pn)
        
        ## return back a phrase representation
        return pidentifier

    def backup(self,wdir,name='hiero_table'):
        """Backup the phrase components 

        :param wdir: the working directory 
        :rtype: None 
        """
        stime = time.time()
        simple_table = os.path.join(wdir,name)
        if os.path.isfile(simple_table+".npz"):
            self.logger.info('Already backed up, skipping...')
            return

        infoa = np.array([self.elen,self.flen],dtype=np.int32) 
        np.savez_compressed(simple_table,self.slist,self.phrases,self.ids,infoa,self.glue)
        self.logger.info('Backed up in %s seconds' % (time.time()-stime))

    @classmethod
    def load_backup(cls,config,name='hiero_table'):
        """Load a backup or instance from file 

        :param config: the global configuration 
        :returns: simple phrase table instance 
        """
        stime = time.time()
        simple_table = os.path.join(config.dir,name+".npz")
        archive = np.load(simple_table)
        slist   = archive["arr_0"]
        phrases = archive["arr_1"]
        ids     = archive["arr_2"]
        info    = archive["arr_3"]
        glue    = archive["arr_4"].item()

        instance = cls(slist,phrases,ids,info[0],info[1],glue)
        instance.logger.info('Loaded backup in %s seconds' % (time.time()-stime))
        return instance         

    cdef HieroRule create_rule(self,input1,input2,lhs,str lang='en'):
        """Query the phrase table for specific phrases

        :param input1: the left side of the phrase 
        :param input2: the right side of the phrase (if exists) 
        :returns: the index of the phrase in the table 
        """
        cdef int[:] alpha = self.slist
        cdef int[:,:] ids = self.ids 
        cdef np.ndarray phrases = self.phrases
        cdef int pidentifier
        cdef unicode left,right,query
        cdef int pn = phrases.shape[0]
        cdef HieroSide side1,side2

        left  = to_unicode(input1.strip())
        right = to_unicode(input2.strip())
        lhs = to_unicode(lhs.strip())
        side1 = HieroSide(left)
        side2 = HieroSide(right)
        query = u"%s ||| %s ||| %s" % (lhs,left,right)

        ## query if the the phrase is a valid one
        pidentifier = _query_table(query, phrases,alpha,pn)

        ## the side identifiers
        if pidentifier != -1: 
            eid = ids[pidentifier][0]
            fid = ids[pidentifier][1]
        else:
            eid = -1; fid = -1
        
        return <HieroRule>HieroRule(lhs,side1,side2,lang=lang,eid=eid,fid=fid,rule_number=pidentifier)
       
cdef class ParaphraseTable(PhraseTableBase):
    """Phrase table for holding paraphrase information"""
    pass

## sparse word pair array

cdef class SparseDictWordPairs(ZubrSerializable):
    """Sparse, dictionary representation of word pairs"""

    def __init__(self,word_dict):
        """Create a SparseDictWordPairs instance

        :param word_dict: the dictionary of word pairs 
        :type word_dict: dict
        """
        self.word_dict = word_dict
        self.logger.info('Sparse word pairs loaded, number of pairs=%d' % len(self.word_dict))
        
    cdef int find_identifier(self,f,e):
        """Find the identifier associated with two pairs

        :param f: the foreign word or word id
        :param e: the english word or word id
        :returns: the global word pair id
        :rtype: int 
        """
        cdef dict wdict = self.word_dict
        return wdict.get((f,e),-1)

    def pfind_identifier(self,f,e):
        cdef dict wdict = self.word_dict
        return wdict.get((f,e),-1)

    @classmethod
    def from_ranks(cls,config):
        """Create a word pair dictionary instance from training ranks 

        :param config: the main configuration, should contain pointer to data
        """
        stime = time.time()
        rank_out,dataset,rank_list = sparse_pairs(config)
        pairs = find_pairs(rank_out,dataset,rank_list)
        instance =  cls(pairs)

        ## log the result 
        instance.logger.info('Found %d pairs in %s seconds' %\
                                 (len(pairs),str(time.time()-stime)))
        return instance

    def backup(self,wdir):
        """Backups the word pairs to file 

        :param wdir: the working directory 
        :rtype: None 
        """
        stime = time.time()
        p_out = os.path.join(wdir,"pairs")
        if os.path.isfile(p_out+".npz"):
            self.logger.info('Already backed up, ignoring')
            return 
        
        np.savez_compressed(p_out,self.word_dict)
        self.logger.info('Backup up in %s seconds' % (time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load an instance from file and backup 

        :param config: the main configuration 
        :returns: SparseDictWordPairs instance
        """
        stime = time.time()
        p_out = os.path.join(config.dir,"pairs.npz")
        archive = np.load(p_out)
        pairs= archive["arr_0"].item()
        instance = cls(pairs)
        instance.logger.info('Loaded backup in %s seconds' % (time.time()-stime))
        return instance

    property size:
        """Returns the size of the word pair dictionary """
        
        def __get__(self):
            """Returns the number of word pairs

            :rtype: int 
            """
            return len(self.word_dict)


cdef class DescriptionWordPairs(SparseDictWordPairs):
    
    @classmethod
    def from_ranks(cls,config):
        """Load from a descripton file 

        :param config: the main configuration 
        """
        pairs = description_pairs(config)
        instance = cls(pairs)
        instance.logger.info('Found %d pairs' % len(pairs))
        return instance
    
    ## backup protocol
    
    def backup(self,wdir):
        """Backups the word pairs to file 

        :param wdir: the working directory 
        :rtype: None 
        """
        stime = time.time()
        p_out = os.path.join(wdir,"description_pairs")
        if os.path.isfile(p_out+".npz"):
            self.logger.info('Already backed up, ignoring')
            return 
        
        np.savez_compressed(p_out,self.word_dict)
        self.logger.info('Backup up in %s seconds' % (time.time()-stime))

    @classmethod
    def load_backup(cls,config):
        """Load an instance from file and backup 

        :param config: the main configuration 
        :returns: SparseDictWordPairs instance
        """
        stime = time.time()
        p_out = os.path.join(config.dir,"description_pairs.npz")
        archive = np.load(p_out)
        pairs= archive["arr_0"].item()
        instance = cls(pairs)
        instance.logger.info('Loaded backup in %s seconds' % (time.time()-stime))
        return instance    

### C METHODS

cdef dict find_pairs(list rank_out,Dataset dataset,np.ndarray rank_list):
    """Find pairs of english-foreign words from an example rank list

    It is a bit slow in python, so I reimplemented here 

    :param rank_out: the rank list file 
    :param dataset: the training dataset 
    :param rank_list: the rank list with items
    """
    cdef int i,j,k,z,rlen,size = dataset.size
    cdef int n,g,dgold,elen,flen
    #cdef unicode line,number,gold_id,number_list,w
    cdef RankPair instance
    cdef int[:] gold,en,rank_item
    cdef set pairs = set()
    cdef list llist
    cdef int ri_len

    for i in range(size):
        line = rank_out[i].strip()
        number,gold_id,number_list = line.split('\t')
        n = int(number); g = int(gold_id)
        llist = [int(w) for w in number_list.split()]
        rlen = len(llist)

        ## data instance
        instance = dataset.get_item(n)
        gold = rank_list[g]
        en = instance.en
        dgold = instance.rep_id
        elen = en.shape[0]
        flen = gold.shape[0]

        ## first the gold item 
        for j in range(flen):
            for k in range(elen):
                pairs.add((gold[j],en[k]))

        ## all the other pairsing 
        for j in range(rlen):
            rank_item = rank_list[llist[j]]
            ri_len = rank_item.shape[0]
            for k in range(ri_len):
                for z in range(elen):
                    pairs.add((rank_item[k],en[z]))

    ## return the dictionary 
    return {o:k for k,o in enumerate(pairs)}

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef int _query_table(unicode query,np.ndarray phrases,int[:] sorted_list,
                          int num_p) except *:
    """Main c method for querying the underlying phrase table using 
    a binary search on the alphanumerically sorted list of phrases. 

    :param phrases: the list of phrases 
    :param sorted_list: the alphabetically sorted indices of above list 
    """
    cdef int start,end,mid,size = num_p - 1
    cdef unicode fragment 

    ## check edge conditions
    if query > phrases[sorted_list[size]]:
        return -1
    if query < phrases[sorted_list[0]]:
        return -1

    start = 0
    end = size
    
    while True:
        mid = (start+end)/2
        fragment = np.unicode(phrases[sorted_list[mid]])

        ## found a match
        if query == fragment:
            return mid
        ## reached a dead end 
        if mid == start == end:
            return -1

        ## split in half 
        if query > fragment:
            start = (mid + 1)
        elif query < fragment:
            end = mid

## FACTORY     

TABLES = {
    "phrasetable" : SimplePhraseTable,
    "hierotable"  : HieroPhraseTable,
    "paratable"   : ParaphraseTable,
}

cpdef PhraseTable(config):
    """Factory method for building a phrase table

    :param config: the global configuration object
    """
    tclass = TABLES.get(config.pt_type,None)
    if not tclass:
        raise ValueError('Unknown phrase table class: %s' % config.pt_type)
    return tclass


## CLI STUFF (called for building phrase tables from phrase files)

def params():
    """The parameters for running the Phrases.pyx module, which can be
    used for building PhraseTable instances from a text file

    """
    groups = {"PhraseTable" : "Settings for building a phrase table"}
    
    options = [
        ("--table_name","table_name","phrase_table.txt","str",
         "The name of the phrase table raw file  [default='phrase_table.txt']","PhraseTable"),
        ("--pt_type","pt_type","phrasetable","str",
         "The type of phrase table object to build  [default='phrasetable']","PhraseTable"),
    ]

    return (groups,options)

def argparser():
    """Returns a configuration for executable models using defaults 

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser    
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    d,options = params()
    argparser = ConfigObj(options,d,description=_heading,version=v)
    return argparser

def main(config):
    """The main execution point for building phrase table objects 

    -- Note: this should not be called directly from ./run_zubr, but should 
    be used in pipelines. 

    :param config: the global configuration object 
    """

    ## try to build model
    load_logger = logging.getLogger('zubr.Phrases.main')

    try:
        pt_class = PhraseTable(config)
        ## build the object
        table = pt_class.from_config(config)
    except Exception,e:
        traceback.print_exc(file=sys.stdout)
        load_logger.error(e,exc_info=True)

    finally:
        ## back up the object
        pass
