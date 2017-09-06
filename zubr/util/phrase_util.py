#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""
import sys
import os
import time
import codecs
import shutil
import gzip
import logging
import numpy as np
from zubr.wrapper.sort import sort_phrase_list
from zubr.Dataset import Dataset
from zubr.util.aligner_util import load_glue_grammar

## utilities for building phrase tables and other things 

__all__ = [
    "preprocess_table",
    "preprocess_hiero",
    "sparse_pairs",
    "description_pairs",
]

ulogger = logging.getLogger('zubr.util.phrase_util')

def __sort_table(path,wdir):
    """Reprin the table to use for sorting 
    
    :param config: the main configuration 
    """
    phrase_table = os.path.join(wdir,"new_phrase_table.txt")
    phrase_order = []
    alpha_sorted = []
    ephrases = {}
    fphrases = {}
    lang_ids = []
    side_ids = []
    
    with codecs.open(path,encoding='utf-8') as my_phrases:
        with codecs.open(phrase_table,'w',encoding='utf-8') as g_phrases: 
            for line in my_phrases:
                try: 
                    left,right,_ = line.split('\t')
                except Exception,e:
                    ulogger.warning('Could not parse the following line (skipping): %s' % line)
                    ulogger.warning(e,exc_info=True)
                    continue
                left = left.strip()
                right = right.strip()
                rep = u"%s ||| %s" % (left,right)
                print >>g_phrases,rep
                phrase_order.append(rep)

                ## store id of each side 
                if left not in ephrases:  ephrases[left] = len(ephrases)
                if right not in fphrases: fphrases[right] = len(fphrases)
                lang_ids.append([ephrases[left],fphrases[right]])

    ## now run the sorting
    sort_phrase_list(phrase_table,wdir)
    
    ## read the new sorted list (note that each number j is actually j - 1)
    new_list = os.path.join(wdir,"phrase_table_ordering.txt")
    ## did not build?
    if not os.path.isfile(new_list):
        ulogger.error(e,exc_info=True)
        raise ValueError('Error sorting the phrase table')

    with codecs.open(new_list) as a_sort:
        for line in a_sort:
            line = line.strip()
            identifier = int(line)-1 ## indices start from 1!
            alpha_sorted.append(identifier)
            side_ids.append(lang_ids[identifier])
            
    ## delete the phrase files
    #os.remove(path)
    #os.remove(phrase_table)

    ## check that the sort indices length matches number of phrases
    if len(alpha_sorted) != len(phrase_order):
        ulogger.error('More indices than phrases!')
        raise ValueError('More indices than phrases!')

    return (
        np.array(alpha_sorted,dtype=np.int32),
        np.array(phrase_order,dtype=np.object),
        np.array(side_ids,dtype=np.int32),
        len(ephrases),
        len(fphrases),
    )

def __sort_hiero(path,wdir):
    """Read the hiero rule file and sort it's contents to create suffix array 

    :param path: the path to the file 
    :param wdir: the working directory
    """
    rules = os.path.join(wdir,"new_hiero_rules.txt")
    phrase_order = []
    alpha_sorted = []
    ephrases = {}
    fphrases = {}
    lang_ids = []
    side_ids = []

    with codecs.open(path,encoding='utf-8') as my_rules:
        with codecs.open(rules,'w',encoding='utf-8') as h_phrases:
            for line in my_rules:
                try:
                    left,right,_ = line.split('\t')
                except Exception,e:
                    ulogger.warning('Could not parse the following line (skipping): %s' % line)
                    ulogger.warning(e,exc_info=True)
                    continue
                left = left.strip()
                right = right.strip()
                rep = u"%s ||| %s" % (left,right)
                print >>h_phrases,rep
                phrase_order.append(rep)

                lright,rright = right.split("|||")
                lright = lright.strip()
                rright = rright.strip()
                if lright not in ephrases: ephrases[lright] = len(ephrases)
                if rright not in fphrases: fphrases[rright] = len(fphrases)
                lang_ids.append([ephrases[lright],fphrases[rright]])

    ## sort the new file
    sort_phrase_list(rules,wdir,name="hiero")

    new_list = os.path.join(wdir,"hiero_table_ordering.txt")
    if not os.path.isfile(new_list):
        ulogger.error(e,exc_info=True)
        raise ValueError('Error sorting the phrase table')

    with codecs.open(new_list) as a_sort:
        for line in a_sort:
            line = line.strip()
            identifier = int(line)-1 ## indices start from 1!
            alpha_sorted.append(identifier)
            side_ids.append(lang_ids[identifier])

    ## remove old files
    #os.remove(path)
    #os.remove(rules)

    ## check that the sort indices length matches number of phrases
    if len(alpha_sorted) != len(phrase_order):
        ulogger.error('More indices than phrases!')
        raise ValueError('More indices than phrases!')

    return (
        np.array(alpha_sorted,dtype=np.int32),
        np.array(phrase_order,dtype=np.object),
        np.array(side_ids,dtype=np.int32),
        len(ephrases),
        len(fphrases),
    )

    
def preprocess_table(config):
    """Build a gzip version of the raw phrase_table, remove the original copy 

    :param config: the main configuration 
    :raises ValueError:
    """
    table_path = os.path.join(config.dir,"phrase_table.txt")

    if not os.path.isfile(table_path):
        #ulogger.error(e,exc_info=True)
        raise ValueError('Cannot find the raw phrase table: %s' % table_path)

    ## sort the phrase list 
    #sort,phrases = __sort_table(table_path,config.dir)
    return __sort_table(table_path,config.dir)

def preprocess_hiero(config):
    """Preprocess the hierarchical rules and create sorted list 

    :param config: the main configuration 
    """
    hiero_path = os.path.join(config.dir,"hiero_rules.txt")
    if not os.path.isfile(hiero_path):
        raise ValueError('Cannot find the hiero table: %s' % hiero_path)

    glue = load_glue_grammar(config)
    return (glue,__sort_hiero(hiero_path,config.dir))

def sparse_pairs(config):
    """Load sparse word pairs from training data 
    
    NOTE: THIS IS PRETTY SLOW 

    :param config: the global pipeline configuration 
    """
    ## load the training data 
    dpath = os.path.join(config.dir,"train.data")
    dataset = Dataset.load(dpath)
    size = dataset.size
    ulogger.info('Loaded dataset for sparse pairs of size=%d' % size)

    ## rank list
    rank_file = os.path.join(config.dir,"ranks.data")
    archive = np.load(rank_file)
    rank_list = archive["arr_0"]

    ## rank output file
    #rank_scored = os.path.join(config.train_ranks)
    #rank_scored = os.path.join(config.train_ranks)
    rank_scored = config.train_ranks

    if not rank_scored or not os.path.isfile(rank_scored):
        raise ValueError('Cannot find rank list file: %s' % str(rank_scored))
    
    ## pair list
    pairs = set()
    ulogger.info('Creating sparse pair instance from ranks...')

    with codecs.open(rank_scored) as my_ranks:
        rline = my_ranks.readlines()
        ulogger.info('my ranks has line size=%d' % len(rline))
        #return (my_ranks.readlines(),dataset,rank_list)
        return (rline,dataset,rank_list)

    ulogger.info('Finished creating pair instances...')


def description_pairs(config):
    """Load pairs from a description file

    :param config: the global configuration file 
    """
    description_file = os.path.join(config.dir,"descriptions.txt")
    if not os.path.isfile(description_file):
        return {}
    pairs = {}

    with codecs.open(description_file,encoding='utf-8') as my_descriptions:
        for line in my_descriptions:
            line = line.strip()
            symbol,word_list = line.split('\t')
            symbol = symbol.strip().lower()
            for word in word_list.split():
                word = word.strip().lower()
                if (symbol,word) in pairs: continue 
                pairs[(symbol,word)] = len(pairs)

    ulogger.info('Created description pairs of size %d' % len(pairs))
    return pairs

def main(config):
    """The main execution point for running the phrase utilities 

    :param config: the global configuration 
    """
    preprocess_table(config)

   
if __name__ == "__main__":
    main(sys.argv[1:]) 

    
