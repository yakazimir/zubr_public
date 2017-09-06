#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementations of common datastructures

"""

import os
import sys 
import logging
import time
import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound,cdivision
from zubr.wrapper.sort import sort_corpus,print_sorted_array
from zubr.ZubrClass cimport ZubrLoggable,ZubrSerializable

## suffix array
ENCODING = 'utf-8'

## module-level logger 
class_builder = logging.getLogger('zubr.Datastructure.ArrayBuilder')


## SPARE ARRAY REPRESENTATION

cdef class Sparse2dArray(ZubrSerializable):
    """Baseclass and datastructure to represent sparse 2d arrays involving pairs of items. 

    One application is for representating translation model parameters, e.g., 
    in Zubr.Alignment. Here one deals with two vocabulary sets, the set of
    source and target words. To define a conditional distribution, the naive 
    solution is to build a 2-d array where you have: | source | * | target |. 
    
    This sparse array decomposes this into three separate arrays: 

    1 : 1-d array representating the source vocab of size | source |
    2 : 1-d representing the target words observed with source (size varies, in sorted order)
    3 : 2-d representing the spans that each source word in array 1 spans in array 2       

    This assumes that words are represented as integer values, perhaps using 
    some independent symbol table. 
    
    Lookup works the following way: 
    
    For a given input pair: id_1 (in source), id_2 (in target), you go to array
    3 to find span within array 2, then do a binary search to find the position of 
    id_2 in 3. This requires O( log | number of target words assoc. w/ id_1 | ), which 
    should be quite fast for highly spare arrays. The index found for 3 will be a unique
    identifier for the pair (id_1, id_2). 

    The class can be customized by implementing a different from_config class method 

    """

    def __init__(self,dim1,dim2,spans):
        """Create a Sparse2dArray instance

        :param dim1: the fist dimension values 
        :param dime2: the second dimension values
        :param spans: the span of second dimension values for each item in dim1
        """
        self.dim1  = dim1
        self.dim2  = dim2
        self.spans = spans

    @cdivision(True)
    @boundscheck(False)
    @wraparound(False)
    cdef int find_id(self,int id1,int id2):
        """Find the unique identifier for two input 

        Note: -1 means that the pair is not known. 

        :param id1: the first index (first dimension)
        :param id2: the second index (the second dimension) 
        :rtype: int 
        """
        cdef int[:] dim1 = self.dim1, dim2 = self.dim2
        cdef int[:,:] spans = self.spans
        cdef int start,end,mid,candidate 

        with nogil:
            ## return unknown 
            if id1 == -1 or id2 == -1: return -1
        
            ## binay search implementation 
            start  = spans[id1][0]
            end    = (spans[id1][1] - 1) ## is this correct?
            if id2 > dim2[end] or id2 < dim2[start]: return -1 
                    
            while True:
                mid = (start + end)/2
                candidate = dim2[mid]

                # ## found 
                if candidate == id2: return mid            
                ## stuck
                if mid == start == end: return -1
                ### gt
                if id2 > candidate: start = (mid + 1)
                ## lt
                if id2 < candidate: end = mid

    property size:
        """The size, or number of unique pairs, in the sparse array"""
    
        def __get__(self):
            """Returns the number of pairs in the array 

            :rtype: int
            """
            return self.dim2.shape[0]


    @classmethod
    def from_config(cls,config):
        """Build from a configuration (not supported in base class)

        :raises: NotImplementedError
        """
        raise NotImplementedError('Not implemented for this class')

    ## backup protocol

    def backup(self,wdir):
        """Back up this model using mostly numpy 

        :param wdir: the working directory 
        :rtype: None 
        """
        raise NotImplementedError 

    def load_backup(self,config):
        """Load the backup created for this instance 

        :param config: the main configuration 
        """
        raise NotImplementedError 

# cdef class SuffixArrayBase:
cdef class SuffixArrayBase(ZubrLoggable):

    def query(self,query_input):
        raise NotImplementedError()

    @classmethod
    def build_array(cls,input_data):
        """build a suffix array instance from a string (or unicode) input

        :param input_data: the input data
        :type input_data: unicode or str
        :returns: suffix array class instance
        """
        raise NotImplementedError()

cdef class StringSuffixArray(SuffixArrayBase):

    """Implementation of a suffix array for string representations

    -- note: just for experimental purposes, for some reason this implementation
    seems to be pretty slow as compared with built-in string operations
    (i.e., for constructing suffixes and querying prefixes)
    """

    def __init__(self,orig_input,prefix_array,sorted_array,size):
        """

        :param orig_input: the (unicode) original input from which the array is built
        :param prefix_array: the array of prefixes
        :param sorted_array: sorted list of prefix_array indices in alphabetical order
        :param size: the size of the input string
        """
        self.orig_input    = orig_input
        self.prefix_array  = prefix_array
        self.sorted_array  = sorted_array
        self.size          = size

    def query(self,query_input):
        """determine to see if there is a substring with prefix equal to query

        :param query_input: the prefix to match
        :rtype: int
        :returns: the index of the first occurence of input_data (or -1)
        """
        data = unicode(query_input,encoding='utf-8')
        return find_str_prefix(data,self.orig_input,self.prefix_array,
                               self.sorted_array,self.size-1)

    @classmethod 
    def build_array(cls,input_data):
        """build a suffix array instance from a string (or unicode) input

        :param input_data: the input data
        :type input_data: unicode or str
        :returns: suffix array class instance
        """
        data = unicode(input_data,encoding='utf-8')
        dsize = len(data)

        ## setup prefixes
        prefixes = np.ndarray((dsize,),dtype=np.int32)
        prefixes[0:,] = range(dsize)

        ## sort the prefix list
        sorted_list = np.zeros((dsize,),dtype=np.int32)
        
        for i in range(dsize):
            binary_insert(prefixes,sorted_list,i,data,dsize)            
        
        return cls(data,prefixes,sorted_list,dsize)

    property array_input:
        """returns the original string input, which is 13the first prefix in the prefix_array"""
        def __get__(self):
            return <unicode>self.prefix_array[0]
        

    property sorted_array:
        """returns the original string input, which is the first prefix in the prefix_array"""
        def __get__(self):
            return <np.ndarray>self.sorted_array

    def __str__(self):
        return self.prefix_array[0]
        
cdef class SentenceSuffixArray(SuffixArrayBase):
    pass

cdef class ListLookupArray(SuffixArrayBase):
    """A Suffix array implementation for querying large lists"""

    @classmethod
    def build_array(cls,input_data):
        pass 
    
cdef class ParallelCorpusSuffixArray(SuffixArrayBase):
    """suffix array for a full, parallel corpus

    -- first attempt to implement a variant of the suffix array datastructure
    described in Callison-Burch, Bannard, Schroeder, ACL 2005

    basic idea:

       -- the entire corpus is used to build the suffix array
       -- suffixes are just words/points word[i]...word[n], 0 <= i <= n, until the end
       -- prefixes are alphabetically sorted (in the normal fashion) for both languages
       -- there is a separate array to keep track of sentences and sentence positions for each word
       -- indexed to each sentence is an alignment string, from which phrases can be extracted
       
    """

    def __init__(self,name,ecorpus,fcorpus,esorted,fsorted,esen,fsen,starts,corpus_len,alignment):
        """

        :param name: the name of the parallel dataset
        :param ecorpus: the full (unicode) english corpus
        :param fcorpus: the full (unicode) foreign corpus
        :param esorted: english sorted prefixes
        :param fsorted: foreign sorted prefixes
        :param esen: the sentence number and position for each english prefix
        :param fsen: the sentence number and position for each foreign prefix
        :param starts: english/foreign positions for beginning of each sentence indexed by sentence
        :param corpus_len: the length of the parallel corpus (i.e, number of sentence)
        """
        self.name    = name
        self.ecorpus = ecorpus
        self.fcorpus = fcorpus
        self.esorted = esorted
        self.fsorted = fsorted
        self.esen    = esen
        self.fsen    = fsen
        self.starts  = starts
        self.size    = corpus_len
        self.alignment = alignment

    @classmethod
    def build_array(cls,name,english_data,foreign_data,alignment,dir='',N=1):
        """build a suffix array instance from parallel corpus input


        :param name: the name of the parallel dataset
        :param english_data: english part of the parallel corpus
        :type english_data: list
        :param foreign_data: foreign part of the parallel corpus
        :type foreign_data: list
        :param alignment: alignment information between both corpora
        :type alignment: list
        :param dir: the working directory
        :type dir: str
        :param N: number of concurrent sorts (for linux only)
        :type N: int
        :returns: CorpusSuffixArray instance
        :rtype: CorpusSuffixArray
        """
        size = len(english_data)
        starts = np.zeros((size,4),dtype=np.int32)

        ### build english data
        class_builder.info('building and preparing the data')
        epre,fpre,eindex,findex,edata,fdata,estr,fstr,epos,fpos\
          = mparallel_data(english_data,foreign_data,size,starts)
          
        elen = epre.shape[0]
        flen = fpre.shape[0]
        e_sorted = np.zeros((elen,),dtype=np.int32)
        f_sorted = np.zeros((flen,),dtype=np.int32)

        ## sort the english prefixes
        class_builder.info('sorting the english suffixes...')
        et = time.time()
        ename = "%s_english" % name

        for i in range(elen):
            suffix_insertion_sort(epre,e_sorted,i,epos,estr,elen)

            ## log progress for very large files
            if i % 50000 == 0:
                class_builder.info('at point %d after %s seconds' % (i,time.time()-et))

        class_builder.info('finished in %s seconds, backing up..' % str(time.time()-et))
        print_sorted_array(e_sorted,dir,ename)

        ## sort the foreign prefixes
        class_builder.info('sorting the foreign suffixes...')
        ft = time.time()
        fname = "%s_foreign" % name

        for i in range(flen):
            suffix_insertion_sort(fpre,f_sorted,i,fpos,fstr,flen)

            ## log progress for very large files
            if i % 50000 == 0:
                class_builder.info('at point %d after %s seconds' % (i,time.time()-ft))

        class_builder.info('finished in %s seconds, backing up..' % str(time.time()-ft))
        print_sorted_array(f_sorted,dir,fname)
        
        return cls(name,edata,fdata,e_sorted,f_sorted,eindex,findex,starts,size,alignment)

    def query(self,input_data,lang='en'):
        """get the index associated with input (if it exists)

        -- note: returns indices where suffixes matching input_data
        are in the same sentence

        :param input_data: the english input
        :type input_data: basestring
        :rtype: int
        :returns: the index of the first occurence of input_data (or -1)
        """
        data = to_unicode(input_data)

        if lang == 'en':
            return find_a_prefix(data,self.ecorpus,self.esorted)
        return find_a_prefix(data,self.fcorpus,self.fsorted)

    def query_english(self,input_data):
        """query the english portion of the data

        :param input_data: the english input
        :type input_data: basestring
        :rtype: int
        :rtype: the index of the first occurency of input_data
        """
        return self.query(input_data,lang='en')

    def query_foreign(self,input_data):
        """query the foreign portion of the data

        :param input_data: the foreign input
        :type input_data: basestring
        :rtype: int
        :rtype: the index of the first occurency of input_data        
        """
        return self.query(input_data,lang='fr')

    ## backup protocol

    def backup(self,wdir,corpus_id='1'):
        """Backup the current model or datastructure to file 

        :param wdir: the working directory 
        :param corpus_id: the id of the corpus
        :rtype: None  
        """
        cout = os.path.join(wdir,str(corpus_id))
        self.logger.info('backing up to: %s' % cout)
        np.savez_compressed(cout,self.ecorpus,self.fcorpus,self.esorted,
                                self.fsorted,self.esen,self.fsen,self.starts,
                                np.array([self.size]),
                                np.array([self.name]),
                                self.alignment)

    @classmethod
    def load_backup(cls,config,corpus_id='1'):
        """Load a backup from file

        :param config: the main configuration 
        """
        cout = os.path.join(config.dir,"%s.npz" % str(corpus_id))
        archive = np.load(cout)
        ecorpus = archive["arr_0"]
        fcorpus = archive["arr_1"]
        esorted = archive["arr_2"]
        fsorted = archive["arr_3"]
        esen    = archive["arr_4"]
        fsen    = archive["arr_5"]
        starts  = archive["arr_6"]
        size    = archive["arr_7"][0]
        name    = str(archive["arr_8"][0])
        a       = archive["arr_9"].tolist()

        return cls(name,ecorpus,fcorpus,esorted,fsorted,esen,fsen,starts,size,a)
        
    ## pickle implementation

    def __reduce__(self):
        return ParallelCorpusSuffixArray,(self.name,self.ecorpus,self.fcorpus,self.esorted,self.fsorted,
                                          self.esen,self.fsen,self.starts,self.size,self.alignment)

    ## accessing suffix array attributes

    property english_data:
        """access the (raw) english unicode data"""
        
        def __get__(self):
            return <np.ndarray>self.ecorpus

    property english_sorted:
        """access the english sorted suffix list"""
        
        def __get__(self):
            return <np.ndarray>self.esorted
        
    property foreign_data:
        """access the (raw) foreign unicode data """
        
        def __get__(self):
            return <np.ndarray>self.fcorpus

    property foreign_sorted:
        """access the foreign sorted suffix list"""
        
        def __get__(self):
            return <np.ndarray>self.fsorted

    property start_pos:
        """the starting positions for each"""

        def __get__(self):
            return <np.ndarray>self.starts

    property esentences:
        """the sentence number and position of each english prefix"""

        def __get__(self):
            return <np.ndarray>self.esen

    property fsentences:
        """the sentence number and position of each foreign prefix"""

        def __get__(self):
            return <np.ndarray>self.fsen

cdef class Indexer:

    """Class for representing start and end spans in sorted suffix arrays"""

    def __init__(self,start=-1,end=-1,size=0):
        """
        
        :param start: the starting point in a given sorted array
        :type start: int
        :param end: the ending point in a given sorted array
        :type end: int
        """
        self.start = start
        self.end   = end
        self.size  = size
        
    property in_array:
        """determine if span is actually in suffix array"""
        def __get__(self):
            if self.start == -1 or self.end == -1:
                return False
            return True
        
### factory

def SuffixArray(object ainput):
    """Factory method for building a suffix array

    :param ainput: input to use to build suffix array
    :type ainput: basestring
    """
    if isinstance(ainput,basestring):
        if ' ' in ainput:
            raise NotImplementedError('sentence array not implemented yet')
        return StringSuffixArray.build_array(ainput)
    raise NotImplementedError()


###############################################
###
### suffix array sorting and query functions


#@boundscheck(False)
#@cdivision(True)
cdef Indexer find_a_prefix(unicode query,np.ndarray data, int[:] ranked):
    """find start finish index of query occurence in suffix-array

    :param query: the fragment to search for
    :param data: the english data
    """
    cdef Indexer cindexer = Indexer()
    cdef int i,start,end,mid
    cdef int qlen = len(query.split())
    cdef unicode fragment
    cdef int size = data.shape[0]
    cdef int lf,rt
    cdef int l,k

    if query > <unicode>(' '.join(data[ranked[size-1]:size])):
        return cindexer

    elif query < <unicode>(' '.join(data[ranked[0]:ranked[0]+qlen])):
        return cindexer

    else:
        start = 0
        end = size

        while True:
            mid = (start+end)/2
            fragment = ' '.join(data[ranked[mid]:ranked[mid]+qlen])

            if query == fragment:
                lf = 1
                rt = 1
                k = mid; l = mid

                ## go left until end
                
                while True:
                    if (mid - lf) < 0:
                        break

                    fragment = ' '.join(data[ranked[mid-lf]:ranked[mid-lf]+qlen])
                    if fragment < query: break
                    k -= 1
                    lf += 1

                ## go right until end
                while True:
                    
                    if (mid + rt) >= size:
                        break

                    fragment = ' '.join(data[ranked[mid+rt]:ranked[mid+rt]+qlen])
                    if fragment > query:
                        break
                    l += 1
                    rt += 1

                cindexer.start = k
                cindexer.end = l
                cindexer.size = qlen
                return cindexer
                                
            if mid == start == end:
                return cindexer

            if query < fragment:
                end = mid

            elif query > fragment:
                start = (mid + 1)
    
#@cdivision(True)
#@boundscheck(False)
cdef int find_str_prefix(unicode query,unicode data,int[:] prefixes,int[:] ranked,int size):
    """determine if prefix is in array using binary search

    :param query: to query to the suffix array
    :param data: the total data to query
    :param prefix: an array of prefixes
    :param ranked: alphabetically sorted prefixes (or pointers to ``prefix``)
    :param size: size of the overall data
    """
    cdef int i
    cdef int start,end,mid
    cdef unicode fragment
    cdef int qlen = len(query)
    cdef int closest
    cdef int lf
    cdef int rt

    ## higher than highest
    if query > data[prefixes[ranked[size]]:prefixes[ranked[size]]+qlen]:
        return -1
    elif query < data[prefixes[ranked[0]]:prefixes[ranked[0]]+qlen]:
        return -1
    
    else:
        start = 0
        end = size

        while True:
            mid = (start+end)/2
            fragment = data[prefixes[ranked[mid]]:prefixes[ranked[mid]]+qlen]

            if query == fragment:
                ## now need to find closest index to 0
                closest = prefixes[ranked[mid]]
                lf = 1
                rt = 1

                ## go left until end of query
                while True:
                    
                    if (mid - lf) < 0:
                        break
                    
                    fragment = data[prefixes[ranked[mid-lf]]:prefixes[ranked[mid-lf]]+qlen]
                    if fragment < query:
                        break
                    
                    if closest > prefixes[ranked[mid-lf]]:
                        closest = prefixes[ranked[mid-lf]]
                        
                    lf += 1

                ## go right until the end of query
                while True:

                    if (mid + rt) >= size:
                        break

                    fragment = data[prefixes[ranked[mid+rt]]:prefixes[ranked[mid+rt]]+qlen]
                    if fragment > query:
                        break

                    if closest > prefixes[ranked[mid-lf]]:
                        closest = prefixes[ranked[mid+rt]]

                    rt += 1

                return closest
                                                
                    
            if mid == start == end:
                return -1

            if query < fragment:
                end = mid

            elif query > fragment:
                start = (mid + 1)
    
#@cdivision(True)
#@boundscheck(False)
cdef void binary_insert(int[:] prefixes,int[:] ranked,int pindex,unicode data,int size):
    """find position of new prefix using binary insertion sort


    :param prefixes: list of prefixes
    :param ranked: current sorted list
    :param pindex: current point in prefix list
    :param data: the actual texual data
    :param size: the size of the textual data
    """
    cdef int i
    cdef int start,end,mid
    cdef unicode fragment
    cdef unicode curr = data[prefixes[pindex]:size]

    if pindex == 0:
        ranked[0] = pindex

    ## rank before top
    elif curr < <unicode>data[prefixes[pindex]:size]:
        ranked[1:] = ranked[0:-1]
        ranked[0] = pindex                        

    ## rank at end
    elif <unicode>data[prefixes[ranked[pindex-1]]:size] < curr:
        ranked[pindex] = pindex

    ## binary search 
    else:
        start = 0
        end = pindex - 1

        while True:
            mid = (start+end)/2
            fragment = data[prefixes[ranked[mid]]:size]

            if mid == start == end:
                ranked[mid+1:] = ranked[mid:-1]
                ranked[mid] = pindex
                break

            if fragment == curr:
                ranked[mid+1:] = ranked[mid:-1]
                ranked[mid] = pindex
                break

            if curr < fragment:
                end = mid

            elif curr > fragment:
                start = (mid + 1)



## !this is really really slow!, better to use either np.argsort() or builtin sorted method
### (np.argsort nd python version however use a huge amount of memory)

## very very slow!

@boundscheck(False)
@cdivision(True)
cdef void suffix_insertion_sort(int[:] prefixes,int[:] ranked,int pindex,
                                list bpos,unicode data,int size):
    """Binary Insertion sort for placing suffixes into sorted list

    -- avoids having to generate all suffixes, which generates too much memory

    :param prefix_str: the full corpus as a unicode string
        
    """
    cdef int i
    cdef int start,end,mid
    cdef unicode fragment
    cdef unicode curr = data[bpos[prefixes[pindex]]:-1]
    cdef unicode first,last

    if pindex == 0:
        ranked[0] = pindex
        return

    elif pindex > 0:

        ## check first 
        first = data[bpos[prefixes[ranked[0]]]:-1]
        if curr < first:
            ranked[1:] = ranked[0:-1]
            ranked[0] = pindex
            return

        ## check last
        last = data[bpos[prefixes[ranked[pindex-1]]]:-1]
        if curr > last:
            ranked[pindex] = pindex
            return 
                
    start = 0
    end = pindex - 1

    while True:
        
        mid = (start+end)/2
        fragment = data[bpos[prefixes[ranked[mid]]]:-1]
        
        if mid == start == end:
            ranked[mid+1:] = ranked[mid:-1]
            ranked[mid] = pindex
            break

        if fragment == curr:
            ranked[mid+1:] = ranked[mid:-1]
            ranked[mid] = pindex
            break

        if curr < fragment:
            end = mid
    
        elif curr > fragment:
            start = (mid + 1)

                
cdef void binary_insert_a(int[:] prefixes,int[:] ranked,int pindex,np.ndarray data,int size):
    """find position of new prefix using binary insertion sort


    :param prefixes: list of prefixes
    :param ranked: current sorted list
    :param pindex: current point in prefix list
    :param data: the actual texual data
    :param size: the size of the textual data
    """
    cdef int i
    cdef int start,end,mid
    cdef unicode fragment
    cdef unicode curr = ' '.join(data[prefixes[pindex]:size])

    if pindex == 0:
        ranked[0] = pindex

    ## rank before top
    elif curr < <unicode>(' '.join(data[prefixes[pindex]:size])):
        ranked[1:] = ranked[0:-1]
        ranked[0] = pindex                        

    ## rank at end
    elif <unicode>(' '.join(data[prefixes[ranked[pindex-1]]:size])) < curr:
        ranked[pindex] = pindex

    ## binary search to find sort position
    
    else:
        start = 0
        end = pindex - 1

        while True:
            mid = (start+end)/2
            fragment = ' '.join(data[prefixes[ranked[mid]]:size])

            if mid == start == end:
                ranked[mid+1:] = ranked[mid:-1]
                ranked[mid] = pindex
                break

            if fragment == curr:
                ranked[mid+1:] = ranked[mid:-1]
                ranked[mid] = pindex
                break

            if curr < fragment:
                end = mid

            elif curr > fragment:
                start = (mid + 1)
                

cpdef tuple mparallel_data(list english,list foreign,int size,int[:,:] starts):
    """create the datastructures for parallel data suffix array

    :param english: the english corpus
    :param foreign: the foreign corpus
    :param fsentences: the foreign sentence positions
    :param size: the size of the corpora
    """
    cdef int i,k
    cdef int english_id = 0
    cdef int foreign_id = 0
    cdef unicode enword,frword
    cdef list ecorpus = []
    cdef list fcorpus = []
    cdef list estarts = [], eends = []
    cdef list fstarts = [], fends = []
    cdef np.ndarray[ndim=2,dtype=np.int32_t] esentences
    cdef np.ndarray[ndim=2,dtype=np.int32_t] fsentences
    cdef np.ndarray[ndim=1,dtype=np.int32_t] eprefixes
    cdef np.ndarray[ndim=1,dtype=np.int32_t] fprefixes
    cdef unicode english_str = u'',foreign_str = u''
    cdef int english_curr = 0,foreign_curr = 0
    cdef list english_pos = [],foreign_pos = []
    cdef int num_ewords = 0, numfwords = 0

    ## take a first pass to calculate the number of words
    class_builder.info('going through the parallel data...')
    
    for i in range(size):

        ## english sentence 
        for k,enword in enumerate(english[i]):
            ecorpus.append(enword)
            estarts.append(i)
            eends.append(k)

            ## keep track of btye position and build corpus string
            english_pos.append(english_curr)
            english_str += enword+u' '
            english_curr += len(enword)+1
            
            ## begin of sentence number ``i``

            if k == 0:
                starts[i][0] = english_id
                starts[i][2] = len(english[i])
            
            english_id += 1

        ## foreign sentence 
        for k,frword in enumerate(foreign[i]):
            fcorpus.append(frword)
            fstarts.append(i)
            fends.append(k)

            foreign_pos.append(foreign_curr)
            foreign_str += frword+u' '
            foreign_curr += len(frword)+1

            ## begin of sentence number ``i``
            if k == 0:
                starts[i][1] = foreign_id
                starts[i][3] = len(foreign[i])
                            
            foreign_id += 1

    ## make english prefix array
    eprefixes = np.zeros((english_id,),dtype=np.int32)
    fprefixes = np.zeros((foreign_id,),dtype=np.int32)
    esentences = np.zeros((english_id,2),dtype=np.int32)
    fsentences = np.zeros((foreign_id,2),dtype=np.int32)

    ## corpora
    ecorpora = np.asarray(ecorpus,dtype=np.unicode)
    fcorpora = np.asarray(fcorpus,dtype=np.unicode)
    
    ## load up english lists
    eprefixes[0:,] = range(english_id)
    esentences[:,1] = eends
    esentences[:,0] = estarts

    ## load up foreign lists
    fprefixes[0:,] = range(foreign_id)
    fsentences[:,1] = fends
    fsentences[:,0] = fstarts

    return (eprefixes,fprefixes,esentences,fsentences,ecorpora,fcorpora,
            english_str,foreign_str,english_pos,foreign_pos)

### tree classes

class Tree:
    pass 
    
class BinaryTree:
    pass 

