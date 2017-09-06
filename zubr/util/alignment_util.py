# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Utilities for working with alignmetn models 

"""
import sys
import os
import logging
import codecs
from collections import Counter
from zubr.util.os_util import make_experiment_directory
import numpy as np
import pickle
import re

util_logger = logging.getLogger('zubr.util.alignment_util')

class AlignerLoaderError(Exception):
    pass  

def copy_settings(config,settings):
    """Create a new config by copying over relevant config field from old config to new

    :param config: the old config 
    :param settings: the new config 
    """
    settings.amax = config.amax
    settings.source = config.source
    settings.target = config.target
    settings.aiters = config.aiters
    settings.aiters2 = config.aiters2
    settings.aiters3 = config.aiters3
    settings.atraining = config.atraining
    settings.aligntraining = config.aligntraining
    settings.aligntesting = config.aligntesting
    settings.emax = config.emax
    settings.maxtree = config.maxtree
    settings.amode = config.amode
    settings.modeltype = config.modeltype
    settings.dir = config.dir
    settings.lower = config.lower
    settings.encoding = config.encoding
    settings.sym = config.sym
    settings.train2 = config.train2
    
def word_map(elex,flex):
    """Create an inverse map from indices to words

    :param elex: the english lexicon map 
    :type elex: dict
    :param flex: the foreign lexicon map 
    :type flex: dict 
    :returns: reversed lexicon map 
    """
    return {
        "e":{identifier:word for (word,identifier) in elex.iteritems()},
        "f":{identifier:word for (word,identifier) in flex.iteritems()}
    }


def output_path(config):
    """Decide where to print alignment output
    
    :param config: the experimental configuration 
    :rtype: str 
    """
    if not config.dir: return ''
    return os.path.join(config.dir,"alignment/test_decode.txt")

def print_table(table,elex,flex,ofile=sys.stdout,minp=0.0001):
        """print the co-occurence probabilities

        :param file: where to the table to
        :type file: file or sys.stdout
        :param elex: the english lexicon map
        :type elex: dict 
        :param flex: the foreign lexicon map 
        :type flex: dict 
        :rtype: None
        """

        words = word_map(elex,flex)
        flen  = table.shape[0]
        elen  = table[0].shape[0]
        util_logger.info("Printing probability table...")

        if isinstance(ofile,basestring):
            out = codecs.open(ofile,encoding='utf-8')
        else:
            out = ofile 

        for sid in range(flen):
            print >>out, "f: %s" % words["f"][sid].encode('utf-8')
            wl = {words["e"][k]:i for k,i in enumerate(table[sid])}
            for w,prob in Counter(wl).most_common():
                if prob < minp or prob <= 0.0: continue
                print >>out,"\t\t%s\t%f" % (w.encode('utf-8'),prob)

        if isinstance(ofile,basestring):
            out.close()

            
def get_numpy_data(path):
    """check if aligner directory contains precompiledd
    numpy array data

    :param path: aligner directory
    :type path: str
    :param source: source extension
    :type source: str
    :param target: target extension
    :type target: str
    :returns: training lexicons and parallel data
    :rtype: tuple 
    :raises: AlignerLoaderError 
    """
    training = os.path.join(path,"train.npy")
    lexs = os.path.join(path,"lex.p") 

    if os.path.isfile(training) and os.path.isfile(lexs):
        try:
            lex = pickle.load(open(lexs))
            t_data = np.load(training)
            return [t_data["arr_0"],t_data["arr_1"],lex[0],lex[1]] 
        except Exception,e:
            raise AlignerLoaderError(e)
    return []

def dump_aligner_data(dir,s,t,lex,name):
    """dump binary aligner data

    :param dir: target dump directory
    :type dir: str
    :param s: source numpy encoded data
    :type s: np.ndarray
    :param t: target numpy encoded data
    :type t: np.ndarray
    :param lex: source and target lexicon
    :type lex: tuple(dict,dict)
    :param name: data name
    :type name: str  
    :rtype: None
    """
    numpy_backup = os.path.join(dir,"%s.npy" % name)
    lex_backup = os.path.join(dir,"lex.p")

    f = open(numpy_backup,"w+b")
    l = open(lex_backup,"w+b")
    try:
        np.savez(f,s,t) 
        pickle.dump(lex,l)
    finally:
        f.close()
        l.close()


def get_lex(align_path):
    """return the lexicon

    :returns: tuple of f and e dict lex
    :rtype: tuple
    """
    lex = os.path.join(align_path,"lex.p")
    if not os.path.isfile(lex):
        raise AlignerLoaderError('missing lex')

    with open(lex) as mylex:
        model_lex = pickle.load(mylex)

    return model_lex

def encode_data(dir,source,target,sdict,tdict,
                stops=False,ignore_oov=True,max_l=100,name="train"):
    """returns np array format of data for running aligner

    :param source: source string
    :type source: unicode
    :param target: target string
    :param sdict: source lexicon map
    :type sdict: dict 
    """
    par_source = []
    par_target = []
    if len(source) != len(target):
        raise AlignerLoaderError('source/target mismatch in size: %d/%d' %\
                                  (len(source),len(target)))

    if dir and name: 
        s_txt = codecs.open(os.path.join(dir,"source.txt"),'w',encoding='utf-8')
        t_txt = codecs.open(os.path.join(dir,"target.txt"),"w",encoding='utf-8')
    
    for k in range(len(source)):
        s = source[k]
        t = target[k]
        if len(s.split())+1 > max_l or len(t.split()) > max_l:
            util_logger.warning('sentence at line: %d exceeds max,skipping' % k) 
            continue
        ## backup in aligner directory

        ## do we want to map these to ``None`` versus -1 and removing? 
        s_ids = [sdict.get(x,None) for x in s.split()]
        t_ids = [tdict.get(x,None) for x in t.split()]

        ## backup the data
        if dir and name:
            print >>s_txt, ' '.join([w for i,w in enumerate(s.split()) if s_ids[i]])
            print >>t_txt, ' '.join([w for o,w in enumerate(t.split()) if t_ids[o]])
        # encode as numpy array 
        ## forgot to add 0! 
        s_ids = np.array([0]+filter(None,s_ids),dtype=np.int32) 
        t_ids = np.array(filter(None,t_ids),dtype=np.int32) 
        par_source.append(s_ids)
        par_target.append(t_ids)
        
    if dir and name:
        t_txt.close()
        s_txt.close()
    ## backup encoded data
    par_source = np.array(par_source,dtype=np.object) 
    par_target = np.array(par_target,dtype=np.object)

    ## backup numpy data if directory exists
    if dir and name: 
        dump_aligner_data(dir,par_source,par_target,(sdict,tdict),name)

    return (par_source,par_target)

def build_lexicon(a_file,encoding='utf-8',lower=False,other=[]):
    """read text files and

    :param a_file: aligner text file
    :type a_file: str
    :param encoding: text encoding
    :type encoding: str
    :param lower: lowercase the training data
    :type lower: bool
    :returns: lexicon and sentence list
    :param other: other files with data to add to lexicon
    :type other: list 
    :rtype: tuple  
    """
    # forgot none 
    lexicon = {u"<EMPTY>":0}
    sentences = []
    
    #try:
    my_data = codecs.open(a_file,encoding=encoding)
    try: 
        for line in my_data:
            line = line.strip()
            if lower:
                line = line.lower()
            sentences.append(line)
            words = line.split()
            for word in words:
                if word not in lexicon:
                    lexicon[word] = len(lexicon)

        ## other stuff to parse/add to lexicon
        for extra in other:
            if not os.path.isfile(extra):
                util_logger.warning('file not found or included: %s' % extra)
                continue

            util_logger.info('assigning unseen words to symbol table...')
            with codecs.open(extra,encoding=encoding) as ex:
                for line in ex:
                    line = line.strip().lower()
                    # if line not in lexicon:
                    #     lexicon[word.lower().strip()] = len(lexicon)
                    for word in line.split():
                        if word not in lexicon:
                            lexicon[word.lower().strip()] = len(lexicon)

    except Exception,e:
        raise AlignerLoaderError(e)

    #finally:
    my_data.close()
    return (lexicon,sentences)

def load_aligner_data(config,sdict={},tdict={}):
    """create alignment dataset to train aligner

    :param config: configuration object
    :type config: zubr.util.config.ConfigAttrs
    :param sdict: source side lexicon
    :type sdict: dict
    :param tdict: target side lexicon
    :type tdict: dict
    :param extra: extra files to be parsed
    :type extra: list 
    :returns: aligner numpy training data 
    :rtype: tuple 
    """
    path = config.atraining
    if path == '' or path is None:
        raise AlignerLoaderError('data path not specified..')

    dir_path = filter(None,path.split('/'))
    name = dir_path[-1]
    prefix = '/'.join(dir_path[:-1])
    exist_t = False
    config.align_dir = ''

    ## back up your dataset/ resulting numpy training files
    if not os.path.isdir(prefix):
        if not os.path.isdir(os.path.join('/',prefix)):
            raise AlignerLoaderError('prefix path not known: %s' % prefix)    
        prefix = os.path.join('/',prefix)

    if config.dir: 
        a_dir = os.path.join(config.dir,"alignment")
    else:
        a_dir = ''


    ## make an alignment directory 
    if config.dir and not os.path.isdir(a_dir):
        make_experiment_directory(a_dir,config=config)
        config.align_dir = a_dir
        util_logger.debug('using alignment directory at: %s' % config.align_dir)
        exist_t = False

    
    elif config.dir and os.path.isdir(a_dir) and config.sym:
        dir2 = os.path.isdir(os.path.join(config.dir,"alignment2"))
        ## switch back
        
        if dir2 and not config.train2:
            config.align_dir = os.path.join(config.dir,"alignment")
            util_logger.debug('using (again) alignment directory at: %s' % config.align_dir)
            exist_t = get_numpy_data(config.align_dir)
        elif dir2 and config.train2:
            config.align_dir = os.path.join(config.dir,"alignment2")
            util_logger.debug('using (again) alignment directory at: %s' % config.align_dir)
            exist_t = get_numpy_data(config.align_dir)
        else:
            config.align_dir = os.path.join(config.dir,"alignment2")
            make_experiment_directory(config.align_dir,config=config)
            util_logger.debug('using alignment directory at: %s' % config.align_dir)
            exist_t = False

    elif config.dir and os.path.isdir(a_dir):
        util_logger.debug('found aligner directory..')
        config.align_dir = a_dir
        util_logger.debug('checking for alignment data...')
        exist_t = get_numpy_data(config.align_dir)            

    else:
        util_logger.warning('not backing up alignment data, no path specified...')


    ## make if it doesn't exist and back up
    
    if not exist_t: 
        util_logger.debug('data not found (or incomplete), building dataset.., (english=%s,foreign=%s)' %\
                      (config.target,config.source))

        source = os.path.join(prefix,"%s.%s" % (name,config.source))
        target = os.path.join(prefix,"%s.%s" % (name,config.target))

        ## check for testing data
        f_test = os.path.join(prefix,"%s_test.%s" % (name,config.source))
        e_test = os.path.join(prefix,"%s_test.%s" % (name,config.target))
        f_val = os.path.join(prefix,"%s_val.%s" % (name,config.source))
        e_val = os.path.join(prefix,"%s_val.%s" % (name,config.target))

        ## don't add unseen words from testing or
        sdict,s_sen = build_lexicon(source,encoding=config.encoding,lower=config.lower,other=[])
        tdict,t_sen = build_lexicon(target,encoding=config.encoding,lower=config.lower,other=[]) 
        util_logger.debug('finished parsing training data, now encoding...')
        source_data,target_data = encode_data(config.align_dir,s_sen,t_sen,
                                              sdict,tdict,stops=config.stops,
                                              ignore_oov=config.ignore_oov,
                                              max_l=config.amax)
    else:
        util_logger.debug('found numpy training data..')
        source_data,target_data,sdict,tdict = exist_t
        
    initial_prob = 1.0/len(tdict)
    if 'sparse' not in config.modeltype.lower():
        table = np.ndarray((len(sdict),len(tdict)),dtype="d")
        table.fill(initial_prob)
    else:
        table = initial_prob
        
    ##buid distortion
    if config.modeltype.lower() == "ibm2":
        distortion = np.ndarray((config.amax,config.amax,config.amax,config.amax),dtype="d")
        distortion.fill(np.inf)
    else:
       distortion = None
              
    return [source_data,target_data,sdict,tdict,table,distortion]


def build_sparse2d(english,foreign,flex):
    """Builds a sparse 2d representatin of parallel dataset

    :param english: the english side of the dataset 
    :param foreign: the foreign side: 
    :param flex: the foreign lexicon 
    """
    fwords = {i:set() for i in flex.values()}
    dsize = english.shape[0]
    span_list  = []
    right_side = []
    
    
    ## find actual pairs in the training data 
    for i in range(dsize):
        english_input = english[i]
        foreign_input = foreign[i]
        elen = english_input.shape[0]
        flen = foreign_input.shape[0]
        for k in range(flen):
            for j in range(elen):
                fwords[foreign_input[k]].add(english_input[j])

    ## make spans
    sorted_list = sorted(flex.values())
    current_start = 0
    for i in sorted_list:
        connections = fwords[i]
        i_sorted_list = sorted(connections)
        for j in i_sorted_list:
            right_side.append(j)
        span_list.append([current_start,current_start+len(connections)])
        current_start = len(right_side)



    assert len(sorted_list) == len(span_list), 'sorted list does not match spans'
    return (
        np.array(sorted_list,dtype=np.int32),
        np.array(right_side,dtype=np.int32),        
        np.array(span_list,dtype=np.int32),
    )      
