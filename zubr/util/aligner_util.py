#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""
import logging
import sys
import os
import re
import pickle
import codecs
import shutil
import numpy as np
from zubr.util._util_exceptions import ZubrUtilError
from zubr.util.os_util import make_experiment_directory 
from zubr.util._util_logger import UtilityLogger

__all__ = [
    "load_aligner_data",
    "encode_data",
    "get_tree_data",
    "get_test_data",
    "build_query_data",
    "get_decoder_data",
    "get_executor_data",
    "build_sparse2d",
]

class AlignerLoaderError(ZubrUtilError):
    pass  

alogger = logging.getLogger('zubr.util.aligner_util')

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
            alogger.warning('sentence at line: %d exceeds max,skipping' % k) 
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
                alogger.warning('file not found or included: %s' % extra)
                continue

            alogger.info('assigning unseen words to symbol table...')
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


def load_other_data(config,dtype='test'):
    """assumes the data already exists"""
    pass 

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

    ## main experiment directory  
    if config.dir and not os.path.isdir(config.dir):
        make_experiment_directory(config.dir,config=config)
        alogger.debug('using experiment directory as: %s' % config.dir)
        
    ## alignment directory
    if config.dir: 
        a_dir = os.path.join(config.dir,"alignment")
    else:
        a_dir = ''

    if config.dir and not os.path.isdir(a_dir):
        make_experiment_directory(a_dir,config=config)
        config.align_dir = a_dir
        alogger.debug('using alignment directory at: %s' % config.align_dir)
        ## check for training data
        #alogger.debug('checking for alignment data...')
        #exist_t = get_numpy_data(config.align_dir)
        exist_t = False

    ## symmetric aligner 2
    elif config.dir and os.path.isdir(a_dir) and config.sym:
        dir2 = os.path.isdir(os.path.join(config.dir,"alignment2"))
        ## switch back
        if dir2:
            config.align_dir = os.path.join(config.dir,"alignment")
            alogger.debug('using (again) alignment directory at: %s' % config.align_dir)
            exist_t = get_numpy_data(config.align_dir)
        else:
            config.align_dir = os.path.join(config.dir,"alignment2")
            make_experiment_directory(config.align_dir,config=config)
            alogger.debug('using alignment directory at: %s' % config.align_dir)
            exist_t = False
            #exist_t = get_numpy_data(config.align_dir)
            
    elif config.dir and os.path.isdir(a_dir):
        alogger.debug('found aligner directory..')
        config.align_dir = a_dir
        alogger.debug('checking for alignment data...')
        exist_t = get_numpy_data(config.align_dir)
    else:
        alogger.warning('not backing up alignment data, no path specified...')

    if not exist_t: 
        alogger.debug('data not found (or incomplete), building dataset.., (english=%s,foreign=%s)' %\
                      (config.target,config.source))

        source = os.path.join(prefix,"%s.%s" % (name,config.source))
        target = os.path.join(prefix,"%s.%s" % (name,config.target))

        ## unseen.txt : a list of potentially unseen foreign components, add to lexicon to have identity
        # orig_data = os.path.join(config.dir,'orig_data') 

        # if os.path.isdir(orig_data):
        #     unseens = os.path.join(orig_data,"unseen.txt")
        # else:
        #     unseens = os.path.join(config.dir,"unseen.txt")
        
        ## check for testing data
        f_test = os.path.join(prefix,"%s_test.%s" % (name,config.source))
        e_test = os.path.join(prefix,"%s_test.%s" % (name,config.target))
        f_val = os.path.join(prefix,"%s_val.%s" % (name,config.source))
        e_val = os.path.join(prefix,"%s_val.%s" % (name,config.target))

        ## don't add unseen words from testing or
        sdict,s_sen = build_lexicon(source,encoding=config.encoding,lower=config.lower,other=[])
        tdict,t_sen = build_lexicon(target,encoding=config.encoding,lower=config.lower,other=[]) 
        alogger.debug('finished parsing training data, now encoding...')
        source_data,target_data = encode_data(config.align_dir,s_sen,t_sen,
                                              sdict,tdict,stops=config.stops,
                                              ignore_oov=config.ignore_oov,
                                              max_l=config.amax)
    else:
        alogger.debug('found numpy training data..')
        source_data,target_data,sdict,tdict = exist_t

    initial_prob = 1.0/len(tdict)
    if 'sparse' not in config.modeltype:
        table = np.ndarray((len(sdict),len(tdict)),dtype="d")
        table.fill(initial_prob)
    else:
        table = initial_prob
    
    ##buid distortion
    if config.modeltype == "IBM2":
        distortion = np.ndarray((config.amax,config.amax,config.amax,config.amax),dtype="d")
        distortion.fill(np.inf)
    else:
       distortion = None
       
    return [source_data,target_data,sdict,tdict,table,distortion]

def build_query_data(config,fdict,edict):
    """build the aligner data for query

    :param config: configuration 
    """
    rank_list = os.path.join(config.dir,"rank_list.txt")
    rank_list_uri = os.path.join(config.dir,"rank_list_uri.txt")

    if not os.path.isfile(rank_list) or \
      not os.path.isfile(rank_list_uri):
        raise AlignerLoaderError('missing rank query data..')

    rank_items = []
    uri_items = []
    uri = [i.strip().split('\t') for i in codecs.open(rank_list_uri,encoding=config.encoding)]

    with codecs.open(rank_list,encoding=config.encoding) as ranks:
        for k,line in enumerate(ranks):
            line = line.strip()
            if config.lower:
                line = line.lower()
            words = [0]+[fdict.get(w,-1) for w in line.split()]
            items = np.array(words,dtype=np.int32)
            rank_items.append(items)
            uri_items.append((uri[k][0].encode(config.encoding),
                             uri[k][1].encode(config.encoding)))

    rank_items = np.array(rank_items,dtype=np.object)
    return (rank_items,uri_items)

def load_glue_grammar(config):
    """loads a simple grammar for finding hiero rules"""
    #grammar_path = os.path.join(config.dir,"grammar.txt")
    apath = '/'.join(config.atraining.split('/')[:-1])
    grammar_path = os.path.join(apath,"grammar.txt")
    if not os.path.isfile(grammar_path):
        raise AlignerLoaderError('missing glue grammar...')

    grammar_table = {}
    
    with codecs.open(grammar_path,encoding='utf-8') as myg:
        for line in myg:
            line = line.strip()
            if re.search(r'^\#',line) or not line: continue
            left,right = line.split(' -> ')
            rhs = tuple([i.strip() for i in right.split()])
            grammar_table[rhs] = left.strip()

    return grammar_table

def get_tree_data(config,ex='',tset="train"):
    """get the tree data for the tree position aligner

    :param config: main alignment configuration object
    :param tset: read trees to training or testing/validation 
    """
    wdir = os.path.join(config.dir,config.atraining)
    if tset == "train": 
        trees = "%s.tree" % wdir
    elif tset == "test":
        trees = "%s_test.tree" % wdir
    elif tset == "validation":
        trees = "%s_valid.tree" % wdir
    elif tset == "rank":
        if ex:
            ex = "%s/rank_list.tree" % ex
        else:
            ex = "rank_list.tree"
        trees = os.path.join(config.dir,ex)
    else:
        AlignmentLoaderError(
            'unknown tree type: %s' % tset)

    data = []
    max_tree = 0.0
    with codecs.open(trees,encoding=config.encoding) as tree:
        for line in tree:
            tseq,tlen = line.split('\t')
            tlen = int(tlen) 
            tseq = [0]+[int(s) for s in tseq.split()]
            if (tset == "train" or tset == "rank") and tlen > max_tree:
                max_tree = tlen
            data.append(np.array(tseq+[tlen-1],dtype=np.int32))
    config.treemax = max_tree
    return np.array(data,dtype=np.object)


def get_test_data(config,fdict,edict):
    """load parallel aligner testing data

    :param config: aligner configuration object
    :param fdict: f side word-id map
    :param edict: e side word-id map
    """
    path = config.atraining
    dir_path = filter(None,path.split('/'))
    name = dir_path[-1]
    prefix = os.path.join('/','/'.join(dir_path[:-1]))
    etest_data = os.path.join(prefix,"%s_test.e" % name) 
    ftest_data = os.path.join(prefix,"%s_test.f" % name)
    tree_out = ''

    if config.modeltype == "treemodel":
        #tree_data = os.path.join(prefix,"%s_test.tree" % name)
        tree_out = get_tree_data(config,tset='test') 

    _,f_sen = build_lexicon(ftest_data,encoding=config.encoding,lower=config.lower)
    _,e_sen = build_lexicon(etest_data,encoding=config.encoding,lower=config.lower)
    f_data,e_data = encode_data(config.align_dir,f_sen,e_sen,fdict,edict,max_l=config.amax+10,
                                name=None)
    return (f_data,e_data,tree_out)
        
def __print_rank_file(dir,ranks,base):
    """Prints the rank list representations (for later use)

    :param dir: the working directory 
    :param rank: the rank information list
    """
    bname = base.replace('.txt','').strip()
    file_out = os.path.join(dir,"encoded_%s.txt" % bname)

    with codecs.open(file_out,'w',encoding='utf-8') as o:
        for k,pair in enumerate(ranks):
            print >>o,"%s\t%s\t%s" %\
              (k,pair[0],' '.join([str(i) for i in pair[1]]))

def get_rdata(config,fdict,edict,ttype="test",poly=False):
    """Encode ranking data as numpy array and the testing data

    :param config: aligner configuration
    :type config: zubr.util.config.ConfigAttrs
    :param fdict: id lookup for foreign words
    :type fdict: dict
    :param edict: id lookup for english words
    :type edict: dict
    :returns: tuple of rank data plus testing data
    :rtype: tuple
    :raises: AlignerLoaderError 
    """
    
    ## get data
    if not config.atraining:
        raise AlignerLoaderError('no alignment data specified')

    if ttype == "test":
        et = "%s%s" % (config.atraining,"_test.e")
        ft = "%s%s" % (config.atraining,"_test.f")
        if poly:
            pl = "%s%s" % (config.atraining,"_test.language")
        
    elif ttype == "train":
        et = "%s%s" % (config.atraining,".e")
        ft = "%s%s" % (config.atraining,".f")
        if poly:
            pl = "%s%s" % (config.atraining,".language")
        
    elif ttype == "valid":
        et = "%s%s" % (config.atraining,"_val.e")
        ft = "%s%s" % (config.atraining,"_val.f")
        if poly:
            pl = "%s%s" % (config.atraining,"_val.language")
        
    else:
        raise ValueError('Uknown evaluation set!: %s' % ttype)

    if (not os.path.isfile(et)) or (not os.path.isfile(ft)):
        raise AlignerLoaderError('alignment test data not found..: %s,%s' % (et,ft)) 
    if (not config.rfile) or (not os.path.isfile(config.rfile)):
        raise AlignerLoaderError('error loading rank file: %s' % str(config.rfile))


    ## poly data
    if poly and not os.path.isfile(pl):
        raise AlignerLoaderError('Cannot find language file: %s' % pl)
    elif poly:
        ## read the poly file here 
        language_identifiers = [l.strip() for l in codecs.open(pl,encoding='utf-8').readlines()]

    lookup = {}
    rank_list = []
    freq_list = []
    full_rank_rep = []
    languages = []

    # rank data 
    with codecs.open(config.rfile,encoding=config.encoding) as rf:

        for k,line in enumerate(rf):
            line = line.strip()
            ## lower case 
            if config.lower:
                line = line.lower()
            words = [0]+[fdict.get(w.strip(),-1) for w in line.split()]

            ## take out empty word 
            flen = float(len(words))
            wfreq = {w:1.0 for w in words}
            ## word counts
            # if config.ffreq:
            #     for word in words:
            #         wfreq[word] += 1.0

            word_freq = [wfreq[w]/flen for w in words]
            items = np.array(words,dtype=np.int32)
            rank_list.append(items)
            freq_list.append(np.array(word_freq,dtype='d'))
            lookup[line] = k
            full_rank_rep.append((line,words))

    ### print encoded rank representations
    if config.dir:
        base = os.path.basename(config.rfile)
        __print_rank_file(config.dir,full_rank_rep,base)

    ## test data
    en_input = []
    en_original = []
    gold_ids = []
    with codecs.open(ft,encoding=config.encoding) as fd:
        with codecs.open(et,encoding=config.encoding) as ed:
            english_data = ed.readlines()
            foreign_data = fd.readlines()
            data_len = len(english_data) 
            if len(english_data) != len(foreign_data):
                raise AlignerLoaderError('test data size doesnt match..')

            for k in range(data_len):
                e = english_data[k].strip()
                #en_original.append(e)
                f = foreign_data[k].strip()
                if config.lower:
                    e = e.lower()
                    f = f.lower()
                    
                ## get rid of stuff uunder a certain threshold (as in chrupala)
                # if len(e.split()[2:]) <= 1:
                #     alogger.warning('testing sentence #%d skipped (too short)' % k)
                #     continue
                if not e.strip() or len(e.split()) <= 1 or not f:
                    alogger.warning('testing sentence #%d skipped (too short, <= 1)' % k)
                    continue

                ## polyglot information 
                if poly:
                    languages.append(language_identifiers[k])
                    
                en_original.append(e)                    
                e = e.split()
                # remove repeating words, order not important
                #     if config.modeltype.lower() == 'ibm1':
                #         repeats = []
                #         total = []
                #         for fw in f.split():
                #             if fw in repeats:
                #                 continue
                #             else:
                #                 repeats.append(fw) 
                #                 total.append(fw) 
                    #     e = set(e)
                    # else:
                    #     alogger.warning('trying to mix word order, not allowed with this model!!')
                
                ## assigns -1 to unknown words
                en = np.array([edict.get(w,-1) for w in e],dtype=np.int32)
                en_input.append(en) 
                if f not in lookup:
                    raise AlignerLoaderError('gold output not known: %s, in file: %s, poly=%s' %\
                                                 (f,config.rfile,str(poly)))
                fid = lookup[f]
                gold_ids.append(fid)

    # make a unicode numpy lookup array
    in_order = range(len(lookup))
                
    for item,k in lookup.items():
        in_order[k] = item

    in_order = np.array(in_order,dtype=np.unicode)
    rank_list = np.array(rank_list,dtype=np.object)
    en_input = np.array(en_input,dtype=np.object)
    gold_ids = np.array(gold_ids,dtype=np.int32)
    freq_list = np.array(freq_list,dtype=np.object)
    eng = np.array(en_original,dtype=np.unicode)

    if not poly: 
        return (rank_list,(en_input,gold_ids),in_order,freq_list,eng)
    assert len(languages) == en_input.shape[0],"language identifiers not right"
    languages = np.array(languages,dtype=np.unicode)
    return (rank_list,(en_input,gold_ids),in_order,freq_list,eng,languages)

def get_decoder_data(config,fdict,edict,ttype='valid',poly=False):
    """Get data for word decoder 


    :param config: the main configuration 
    :param fdict: the foreign lexicon 
    :param edict: the english lexicon
    :param ttype: the type of data 
    """
    config.lower = True

    ## if eval_set = train, then replace _bow data (don't want to decode other stuff)
    if ttype == "train":
        alogger.info('Re-arranging data for train eval: %s' % config.atraining)
        eorig = config.atraining+".e"
        forig = config.atraining+".f"
        ebow = config.atraining+"_bow.e"
        fbow = config.atraining+"_bow.f"
        ebackup = config.atraining+"_orig.e"
        fbackup = config.atraining+"_orig.f"
        
        # ebow = os.path.join(config.atraining,"_bow.e")
        # fbow = os.path.join(config.atraining,"_bow.f")
        # ebackup = os.path.join(config.atraining,"_orig.e")
        # fbackup = os.path.join(config.atraining,"_orig.f")
        
        ## use the bow file 
        if os.path.isfile(ebow) and os.path.isfile(fbow):
            alogger.info('Found bow data, making this main set')
            shutil.copy(eorig,ebackup); shutil.copy(forig,fbackup)
            os.remove(eorig); os.remove(forig)
            shutil.copy(ebow,eorig); shutil.copy(fbow,forig)
        else:
            alogger.info('Not found: ebow=%s, fbow=%s' % (ebow,fbow))
            
    
    if poly: 
        rl,inp,order,freq,enorig,langs = get_rdata(config,fdict,edict,ttype=ttype,poly=True)
    else:
        rl,inp,order,freq,enorig = get_rdata(config,fdict,edict,ttype=ttype,poly=False)

    rlsize = rl.shape[0]
    en,gid = inp
    rank_map = {}

    for i in range(rlsize):
        rvalue = order[i]
        assert len(rvalue.split())+1 == rl[i].shape[0],"wrong rvalue"
        rank_map[order[i]] = i

    ## create a rank list map for looking up sequences
    if poly:
        return (en,enorig,rank_map,gid,langs)
    return (en,enorig,rank_map,gid)

def get_executor_data(config,fdict,edict,ttype='valid',poly=False):
    """Get data for running executable model 

    :param config: the main experiment configuration 
    :param fdict: the foreign dictionary 
    :param edict: the english dictionary/lexicon 
    :param ttype: the type of data to evaluate on 
    :param poly: whether the dataset is a polyglot model
    """
    rank_items = {}
    config.lower = True

    ## make a rank file for dataset of interest
    rank_list     = os.path.join(config.dir,"rank_list.txt")
    config.rfile = rank_list

    #if not os.path.isfile(rank_list):
    if not os.path.isfile(rank_list):
    
        #alogger.info('Building new rank list, current one not found')

        ## the known outputs 
        ftest     = config.atraining+"_test.f"
        fvalid    = config.atraining+"_val.f"
        ftrain    = config.atraining+".f"
        out_data  = [ftest,fvalid,ftrain]

        with codecs.open(rank_list,'w',encoding='utf-8') as my_ranks: 
            for data_type in out_data:
                if not os.path.isfile(data_type):
                    continue
                with codecs.open(data_type,encoding='utf-8') as my_file:
                    for line in my_file:
                        line = line.strip().lower()
                        if line not in rank_items:
                            print >>my_ranks,line
                            rank_items[line] = len(rank_items)

    else:
        alogger.info('Rank list already built, skipped...')


    ##
    ## if eval_set = train, then replace _bow data (don't want to decode other stuff)
    if ttype == "train":
        alogger.info('Re-arranging data for train eval: %s' % config.atraining)
        eorig = config.atraining+".e"
        forig = config.atraining+".f"
        ebow = config.atraining+"_bow.e"
        fbow = config.atraining+"_bow.f"
        ebackup = config.atraining+"_orig.e"
        fbackup = config.atraining+"_orig.f"
        
        # ebow = os.path.join(config.atraining,"_bow.e")
        # fbow = os.path.join(config.atraining,"_bow.f")
        # ebackup = os.path.join(config.atraining,"_orig.e")
        # fbackup = os.path.join(config.atraining,"_orig.f")
        
        ## use the bow file 
        if os.path.isfile(ebow) and os.path.isfile(fbow):
            alogger.info('Found bow data, making this main set')
            shutil.copy(eorig,ebackup); shutil.copy(forig,fbackup)
            os.remove(eorig); os.remove(forig)
            shutil.copy(ebow,eorig); shutil.copy(fbow,forig)
        else:
            alogger.info('Not found: ebow=%s, fbow=%s' % (ebow,fbow))

    # ## now build the datasets
    # config.lower = True
    if not poly:
        rl,inp,order,freq,enorig = get_rdata(config,fdict,edict,ttype=ttype,poly=False)
    else:
        rl,inp,order,freq,enorig,langs = get_rdata(config,fdict,edict,ttype=ttype,poly=poly)
    en,gid = inp

    ## remake the rank list just to be careful that the order hasn't changed
    rlsize = rl.shape[0]
    rank_map = {}

    for i in range(rlsize):
        rvalue = order[i]
        assert len(rvalue.split())+1 == rl[i].shape[0],"wrong rvalue"
        rank_map[order[i]] = i

    if not poly:
        return (en,enorig,rank_map,gid,order)
    return (en,enorig,rank_map,gid,order,langs)

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
        current_start = len(span_list)

    return (
        np.array(sorted_list,dtype=np.int32),
        np.array(right_side,dtype=np.int32),        
        np.array(span_list,dtype=np.int32),
    )           

