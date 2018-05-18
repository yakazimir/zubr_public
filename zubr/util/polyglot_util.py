# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package. 

author : Kyle Richardson

utilities for building ``polyglot`` models

"""

import os
import re
import random
import codecs
import shutil
import operator
import functools
import logging
from copy import deepcopy
from collections import defaultdict
from shutil import copytree,copy
from zubr.util.learn_bpe import from_dictionary
from zubr.util.apply_bpe import segment_data,segment_sem
from zubr.util.latin_encoding import short_encode, small_encode

__all__ = [
    "read_data_directory",
    "swap_results",
]

util_logger = logging.getLogger('zubr.util.polyglot_util')
    
def __univeral_component(raw_component):
    """A universal tokenizer for components

    :param raw_component: the raw/original component representation
    """
    pass


def __run_bpe(config,elex,name='codes'):
    """Run the bpe code for segmenting the polyglot data (if selected)

    :param config: the main configuration 
    :param elex: the english/source side lexicon 
    """
    ## make new directory for bpe 
    new_dir = os.path.join(config.dir,"bpe")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    ## make file for codes, run bpe learn
    util_logger.info('Running the BPE algorithm (might take a while)...')
    codes = os.path.join(new_dir,"%s.txt" % name)
    from_dictionary(elex,codes)

    
    ## print dictionary
    # dictionary = os.path.join(new_dir,"vocab.txt")
    # with codecs.open(dictionary,'w',encoding='utf-8') as vfile:
    #     for word,freq in elex.items():
    #         print >>vfile,"%s\t%d" % (word,int(freq))


def __freq_compound_split(components):
    """Tries to split compounds using a frequency based approach 

    -- Note: limits sub components to 3 

    :param components: the foreign component vocabulary maps
    """
    new_compound = {}

    for (foreign,frequency) in components.items():
        
        fwords = foreign.split()
        for word in fwords:
            base_occ = components[word]
            top = base_occ
            best = word
            wordlen = len(word)
            if wordlen <= 4 and base_occ > 1: continue

            #look for up to three word splits 
            for start in range(wordlen):
                for mid in range(start+1,wordlen-1):
                    candidate =  word[0:start]+" "+word[start:mid]+" "+word[mid:wordlen]
                    candidate = re.sub(r'\s+', ' ',candidate).strip()
                    clen = len(candidate.split())
                    ccounts = [components.get(c,0) for c in candidate.split()]
                    small = [p for p in candidate.split() if len(p) <= 2]
                    if small: continue
                    cscore = float(functools.reduce(operator.mul,ccounts,1))**(1.0/float(clen))
                    if cscore > top:
                       best = candidate

            if best != word:
                new_compound[word] = best

    
    return new_compound


## BPE segmentation algorithm

def __get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def __merge_vocab(pair,v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

                
def __read_data(lang_list,names,over_sample=False,maxs=None,sub_word=False,small=False):
    """Read the full data
    
    :param lang_list: the list of languages to parse 
    :type lang_list: list
    """
    word_occurrence = defaultdict(int)
    reduced_vocab = defaultdict(int) 
    component_occurrence = defaultdict(int)
    parallel_data = []
    dataset_count = defaultdict(int)
    lang_positions = {}

    ## go through each dataset
    for (path,name) in lang_list:
        if not os.path.isdir(path): continue

        file_name = names.get(name,name)
        e_data = os.path.join(path,"%s_bow.e" % file_name)
        f_data = os.path.join(path,"%s_bow.f" % file_name)

        ## actual data
        english = codecs.open(e_data,encoding='utf-8').readlines()
        foreign = codecs.open(f_data,encoding='utf-8').readlines()

        ## tree file
        tree_file = os.path.join(path,"%s.poly_tree" % file_name)
        tree_file2 = os.path.join(path,"%s.tree" % file_name)
        try:   trees = codecs.open(tree_file,encoding='utf-8').readlines()
        except:trees = codecs.open(tree_file2,encoding='utf-8').readlines()

        data_len = len(english)

        ## check that they have the same length 
        assert len(english) == len(foreign), "Data mismatch size: %s" % name
        lang_positions[name] = [len(parallel_data),None]

        for data_point in range(data_len):

            dataset_count[name] += 1
            
            english_side = english[data_point].strip().lower()
            foreign_side = foreign[data_point].strip().lower()

            ## get the tree information 
            tree = trees[data_point].strip()
            #tree_data.append(tree)
            parallel_data.append((name,english_side,foreign_side,tree))

            ## english words 
            for eword in english_side.split():
                eword = eword.lower().strip()
                word_occurrence[eword] += 1
                
                ## reduce 
                if small:
                    reduced = small_encode(eword)
                    reduced_vocab[reduced] += 1
                    
            ## component words
            for fword in foreign_side.split():
                fword = fword.lower().strip()
                component_occurrence[fword] += 1

        lang_positions[name][-1] = len(parallel_data)

    ## over sample the dataset?
    if over_sample:
        to_match = max(dataset_count.value()) if maxs is None else maxs
        random.seed(42)
        extra = []

        for lang,count in dataset_count.items():
            if count >= to_match: continue
            ## add random examples
            lang_total = count

            #while True:
            indices = range(lang_positions[lang][0],lang_positions[lang][1])
            random.shuffle(indices)

            while True:
                
                if lang_total >= to_match: break
                for item in indices:
                    if lang_total >= to_match: continue 
                    extra.append(parallel_data[item])
                    lang_total += 1

        ## add extra data points 
        parallel_data += extra

    reduced = reduced_vocab if small else deepcopy(word_occurrence)
    return (parallel_data,word_occurrence,reduced,component_occurrence)

def __generate_train(train_data,lang_loc,transforms,wdir,pseudolex,lang_ids,aclasses,extra_data):
    """Generates the train data and rank lists

    :param train_data: the overall parallel training data 
    :param lang_list: the language list 
    :param transforms: the compound transformations (if any)
    :param wdir: the working directory
    """
    train_path     = os.path.join(wdir,"train")
    held_out_path  = os.path.join(wdir,"held_out")
    rank_path      = os.path.join(wdir,"ranks")

    ## generate directories 
    if not os.path.isdir(train_path): os.mkdir(train_path)
    if not os.path.isdir(held_out_path): os.mkdir(held_out_path)
    if not os.path.isdir(rank_path): os.mkdir(rank_path)

    # edata = os.path.join(wdir,"polyglot.e")
    # fdata = os.path.join(wdir,"polyglot.f")
    # lidentifiers = os.path.join(wdir,"polyglot.languages")

    edata = os.path.join(train_path,"polyglot.e")
    fdata = os.path.join(train_path,"polyglot.f")
    epseudo = os.path.join(train_path,"polyglot_bow.e")
    fpseudo = os.path.join(train_path,"polyglot_bow.f")
    lidentifiers = os.path.join(train_path,"polyglot.language")
    global_rank = os.path.join(rank_path,"global_rank_list.txt")
    polyglot_tree = os.path.join(train_path,"polyglot.tree")
    rank_tree = os.path.join(rank_path,"rank_list.tree")
    rank_classes = os.path.join(rank_path,'rank_list_class.txt')
    ## extra pairs
            
    pseudolex = set()
    ## pseudo lexicons
    data_lex = set()
    tree_map = {}
    pseudolex_files = False
    extra_pairs = set()

    ## read the pseudo lexicon files for all languages 
    for (path,_) in lang_loc:
        if not os.path.isdir(path): continue 
        
        pseudo_file = os.path.join(path,"pseudolex.txt")
        if os.path.isfile(pseudo_file):
            pseudolex_files = True
            with codecs.open(pseudo_file,encoding='utf-8') as lex:
                for line in lex:
                    line = line.strip()
                    data_lex.add(tuple([p.strip() for p in line.split('\t')]))

        ## extra descriptions
        extra_file = os.path.join(path,"extra_pairs.txt")
        if os.path.isfile(extra_file):
            util_logger.info('Found extra pairs: %s' % extra_file)
            with codecs.open(extra_file,encoding='utf-8') as extra:
                for line in extra:
                    line = line.strip()
                    try: 
                        sem,description = line.split('\t')
                        extra_pairs.add((sem,description.strip().lower()))
                    except ValueError:
                        pass 

    ## training data : single training data 
    with codecs.open(edata,'w',encoding='utf-8') as e:
        with codecs.open(fdata,'w',encoding='utf-8') as f:
            with codecs.open(lidentifiers,'w',encoding='utf-8') as ids:
                with codecs.open(epseudo,'w',encoding='utf-8') as ep:
                    with codecs.open(fpseudo,'w',encoding='utf-8') as fp:
                        with codecs.open(polyglot_tree,'w',encoding='utf-8') as t:
                            for (name,nl,sem,tree) in train_data:
                                final_rep = ' '.join([transforms.get(w,w) for w in sem.split()])
                                print >>ids, "<!%s!>" % name.strip() #lang_ids[name][0]

                                ## main data 
                                print >>e,nl.strip()
                                print >>f,final_rep.strip()
                                tree_map[final_rep.strip()] = tree

                                ## bow data for evaluation
                                print >>fp,"<!%s!> %s" % (name,final_rep.strip())
                                print >>ep,nl.strip()

                                ## tree print
                                print >>t,tree.strip()
                            
                                for word in final_rep.split():
                                    pseudolex.add(word.strip())




    #if pseudolex:
    with codecs.open(edata,'a',encoding='utf-8') as e:
        with codecs.open(fdata,'a',encoding='utf-8') as f:

            ## pseudolex files 
            if pseudolex_files: 
                for item in data_lex:
                    if len(item) != 2: continue
                    sem,en = item
                    for _ in range(3):
                        print >>f,sem
                        print >>e, en

            ## pseudolex from data instead
            else:
                for item in pseudolex:
                    for _ in range(3):
                        print >>f,item.lower().strip()
                        print >>e,item.lower().strip()

            ### additional pairs of descriptions
            if extra_pairs:
                for (sem,text) in extra_pairs:
                    print >>f,sem.strip().lower()
                    print >>e,text.strip().lower()

            ## extra pairs?
            if extra_data and os.path.isfile(extra_data):
                util_logger.info('Found extra parallel data: %s' % extra_data)
                with codecs.open(extra_data,encoding='utf-8') as extra:
                    for line in extra:
                        line = line.strip()
                        try: 
                            sem,description = line.split('\t')
                            print >>f,sem.lower().strip()
                            print >>e,description.lower().strip()
                        except ValueError:
                            pass 
                        
    ## rank lists
    
    ## global rank list
    rclass = os.path.join(rank_path,"rank_list_class.txt")

    with codecs.open(global_rank,'w',encoding='utf-8') as globalr:
        with codecs.open(rank_tree,'w',encoding='utf-8') as t:
            with codecs.open(rclass,'w',encoding='utf-8') as rc:
                for (path,name) in lang_loc:
                    rank_file = os.path.join(path,"rank_list.txt")
                    #new_rank = os.path.join(wdir,"rank_list_%s.txt" % name)
                    new_rank = os.path.join(rank_path,"rank_list_%s.txt" % name)
                    try: rank_tree = codecs.open(os.path.join(path,"rank_list.poly_tree")).readlines()
                    except: rank_tree = codecs.open(os.path.join(path,"rank_list.tree")).readlines()

                    ## print new rank file with 
                    with codecs.open(new_rank,'w',encoding='utf-8') as newr:
                        with codecs.open(rank_file,encoding='utf-8') as oldr:
                            for n,line in enumerate(oldr):
                                line = line.strip().lower()
                                final_rep = ' '.join([transforms.get(w,w) for w in line.split()])
                                print >>newr,' '.join([transforms.get(w,w) for w in line.split()])
                                print >>globalr,"<!%s!> %s" % (name,' '.join([transforms.get(w,w) for w in line.split()]))
                                ## print the rank trees
                                print >>t,rank_tree[n].strip()

                                ## find the classes
                                first,_ = rank_tree[n].split('\t')
                                indices = [int(p) for p in first.split()]
                                fun_rep = ' '.join([w for k,w in enumerate(final_rep.split()) if indices[k] == 1])
                                identifier = aclasses.get(fun_rep,-1)
                                class_seq = [identifier if indices[k] == 1 else -1 for k,w in enumerate(final_rep.split())]
                                print >>rc,' '.join([str(i) for i in class_seq])

    epolyglot_test = os.path.join(held_out_path,"polyglot_test.e")
    fpolyglot_test = os.path.join(held_out_path,"polyglot_test.f")
    tpolyglot_lang = os.path.join(held_out_path,"polyglot_test.language")
    
    epolyglot_val = os.path.join(held_out_path,"polyglot_val.e")
    fpolyglot_val = os.path.join(held_out_path,"polyglot_val.f")
    vpolyglot_lang = os.path.join(held_out_path,"polyglot_val.language")
    
    eptest = codecs.open(epolyglot_test,'w',encoding='utf-8')
    fptest = codecs.open(fpolyglot_test,'w',encoding='utf-8')
    epvalid = codecs.open(epolyglot_val,'w',encoding='utf-8')
    fpvalid = codecs.open(fpolyglot_val,'w',encoding='utf-8')
    lpvalid = codecs.open(vpolyglot_lang,'w',encoding='utf-8')
    lptest  = codecs.open(tpolyglot_lang,'w',encoding='utf-8')

    ## testing and validation data
    for (path,name) in lang_loc:
        if not os.path.isdir(path): continue
        file_suffix = lang_ids[name][-1]

        etest = os.path.join(path,"%s_test.e" % file_suffix)
        ftest = os.path.join(path,"%s_test.f" % file_suffix)
        entest = os.path.join(held_out_path,"%s_test.e" % name)
        fntest = os.path.join(held_out_path,"%s_test.f" % name)

        with codecs.open(etest,encoding='utf-8') as etest:
            with codecs.open(entest,'w',encoding='utf-8') as netest:
                for line in etest:
                    line = line.strip()
                    print >>netest,line
                    print >>eptest,line

        with codecs.open(ftest,encoding='utf-8') as ftest:
            with codecs.open(fntest,'w',encoding='utf-8') as nftest:
                for line in ftest:
                    line = line.strip()
                    print >>nftest,' '.join([transforms.get(w,w) for w in line.split()])
                    print >>fptest,"<!%s!> %s" % (name,' '.join([transforms.get(w,w) for w in line.split()]))
                    print >>lptest,"<!%s!>" % name

        evalid = os.path.join(path,"%s_val.e" % file_suffix)
        fvalid = os.path.join(path,"%s_val.f" % file_suffix)
        envalid = os.path.join(held_out_path,"%s_val.e" % name)
        fnvalid = os.path.join(held_out_path,"%s_val.f" % name)

        with codecs.open(evalid,encoding='utf-8') as evalid:
            with codecs.open(envalid,'w',encoding='utf-8') as nevalid:
                for line in evalid:
                    line = line.strip()
                    print >>nevalid,line
                    print >>epvalid,line

        with codecs.open(fvalid,encoding='utf-8') as fvalid:
            with codecs.open(fnvalid,'w',encoding='utf-8') as nfvalid:
                for line in fvalid:
                    line = line.strip()
                    print >>nfvalid,' '.join([transforms.get(w,w) for w in line.split()])
                    print >>fpvalid,"<!%s!> %s" % (name,' '.join([transforms.get(w,w) for w in line.split()]))
                    print >>lpvalid,"<!%s!>" % name
    eptest.close()
    fptest.close()
    epvalid.close()
    fpvalid.close()
    lpvalid.close()
    lptest.close()

def __description_files(languages,wdir):
    """Read the description files (if they exist) for each data and conjoin

    :param languages: the location of the different languages 
    :param wdir: the working directory 
    """
    symbol_word_map = defaultdict(set)

    ## read the descriptions file 
    for (path,name) in languages:
        descriptions = os.path.join(path,"descriptions.txt")
        if not os.path.isfile(descriptions): continue
        with codecs.open(descriptions,encoding='utf-8') as d:
            for line in d:
                line = line.strip()
                symbol,word_list = line.split('\t')
                symbol = symbol.strip().lower()
                for word in word_list.split():
                    word = word.strip().lower()
                    symbol_word_map[symbol.lower()].add(word)

    ## print new descriptions file
    new_descriptions = os.path.join(wdir,"descriptions.txt")
    with codecs.open(new_descriptions,'w',encoding='utf-8') as d:
        for (symbol,wset) in symbol_word_map.items():
            if len(symbol) <= 2: continue
            print >>d,"%s\t%s" % (symbol,' '.join(wset))

    ## segmented descriptions
        
            
def __read_abstract(apath):
    """Read abstract classe file

    :param apath: the path to the abstract file
    """
    classes = {}
        
    with codecs.open(apath,encoding='utf-8') as my_a:
        for k,line in enumerate(my_a):
            line = line.strip()
            for item in line.split():
                ## need to preprocess here 
                rep = re.sub(r'\_',' ',item)
                rep = re.sub(r'([a-z])([A-Z])',r'\1 \2',rep)
                rep = re.sub(r'\-',' ',rep)
                rep = rep.strip().lower()
                classes[rep] = k

    return classes
                
def read_data_directory(config,data_path,names={},data_path2=None,names2={}):
    """Read a directory of datasets an build a single dataset 

    :param config: the main configuration 
    :rtype config: zubr.util.config.ConfigAttrs
    :param data_path: the path to the data 
    :type data_path: str
    :returns: balbalal
    """
    langs = names.keys() if not config.langs else config.langs.split("+")
    lang_ids = {i:(k,names.get(i,i)) for k,i in enumerate(langs)}

    ## the list of datasets
    datasets = [(os.path.join(data_path,i),i) for i in os.listdir(data_path) if i in langs]

    ## secondary data? 
    if data_path2 and names2:
        langs2 = names2.keys()
        datasets2 = [(os.path.join(data_path2,i),i) for i in os.listdir(data_path2) if i in langs2 and '.DS' not in i]
        datasets += datasets2
        names.update(names2)
        lang_ids2 = {i:(k,names.get(i,i)) for k,i in enumerate(langs2)}
        lang_ids.update(lang_ids2)
    
    ## read the training data 
    data,ewords,segmented,fwords = __read_data(datasets,names,config.over_sample,config.max_sample,small=config.small)

    ## read abstract classe file
    if config.abstract_classes and os.path.isfile(config.abstract_classes):
        aclasses = __read_abstract(config.abstract_classes)
    else:
        util_logger.warning('No abstract class file found,path=%s' % config.abstract_classes)
        aclasses = {}

    ## parse description files
    __description_files(datasets,config.dir)
    
    ## compounds transformations
    transforms = {}
    
    ## create rank lists and training data
    __generate_train(data,datasets,transforms,config.dir,config.pseudolex,lang_ids,aclasses,config.more_data)

    config.atraining = os.path.join(os.path.join(config.dir,"train"),"polyglot")
    config.lang_ids = lang_ids

    ## do the bpe stuff on the source side
    if config.sub_word:
        __run_bpe(config,segmented)
        segment_data(config)

    ## do the bpe stuff on the target or semantic side 
    if config.sem_sub_word:
        __run_bpe(config,fwords,name='sem_codes')
        segment_sem(config)

def swap_results(wdir,name):
    """Swap the results file for the current languages

    :param wdir: the working directory
    :param name: the name 
    """
    rank_path = os.path.join(wdir,"results")
    ## make a rank path 
    if not os.path.isdir(rank_path):os.mkdir(rank_path)

    rank_list = os.path.join(wdir,"rank_results.txt")
    nrank_list = os.path.join(rank_path,"rank_results_%s.txt" % name)
    shutil.move(rank_list,nrank_list)
