# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package. 

author : Kyle Richardson

Utilities for working with dataset objects and associated
objects

"""
import os
import codecs
import logging
from collections import defaultdict

util_logger = logging.getLogger('zubr.util.alinger_extractor')

def __return_file_type(work_dir,dtype):
    wtype = 'w'

    #if dtype == 'train':
    if 'train' in dtype:
        path = os.path.join(work_dir,"TRAIN_RESULTS.txt")
    elif dtype == 'valid':
        path = os.path.join(work_dir,"VALID_RESULTS.txt")
    elif dtype == 'test':
        path = os.path.join(work_dir,"TEST_RESULTS.txt")
    elif dtype == 'valid-select':
        path = os.path.join(work_dir,"VALID_SELECT_RESULTS.txt")
    elif 'baseline' in dtype:
        path = os.path.join(work_dir,"rank_results.txt")
    elif 'valid_average' in dtype:
        path = os.path.join(work_dir,"VALID_RESULTS_avg.txt")
    if os.path.isfile(path):
        wtype = 'a'
    return (path,wtype)

def __rank_results_file(wdir):
    """Create a results result directory if not there

    :param wdir: the current working directory 
    """
    ro = os.path.join(wdir,"rank_results")
    if not os.path.isdir(ro):
        os.mkdir(ro)

def __print_score(path,wtype,at1,at2,at5,at10,mrr,descr='BASELINE',ll="?"):
    with open(path,wtype) as my_results:
        if wtype == 'a':
            print >>my_results,""
            print >>my_results,"#"*10
        
        print >>my_results,"results for %s" % (descr)
        print >>my_results,"likelihood=%s" % ll        
        print >>my_results,"="*10
        print >>my_results, "Accuracy @1: %f" % at1
        print >>my_results, "Accuracy @2: %f" % at2
        print >>my_results, "Accuracy @5: %f" % at5
        print >>my_results, "Accuracy @10: %f" % at10
        print >>my_results, "MRR: %f" % mrr
        print >>my_results,'-'*10

def __print_poly_ranks(path,ranks,gold,k=100):
    """Backup the ranks of the polyglot decoders 

    :param path: the working directory path 
    """
    rank_out = os.path.join(path,"ranks.txt")
    total_size = ranks.shape[0]

    with codecs.open(rank_out,'w',encoding='utf-8') as my_ranks:
        for i in range(total_size):
            rank_pos = gold[i]
            rank_value = ranks[i][rank_pos]
            print >>my_ranks,"%d\t%d\t%s" % (i,rank_value,' '.join([str(z) for z in ranks[i]][:k]))
    
def score_rank(workdir,ranks,gold_pos,dtype):
    """Report on a rank score given gold positions

    :param workdir: the working directory 
    :param ranks: the ranks of each item 
    :param gold_pos: the gold locations of items
    :param dtype: the type of data being scored
    :rtype: None
    """
    total_size = ranks.shape[0]
    at_1  = 0.0
    at_2  = 0.0
    at_10 = 0.0
    at_5  = 0.0
    mrr = 0.0

    for i in range(total_size):
        rank_size = ranks[i].shape[0]
        rank_pos = gold_pos[i]
        if rank_pos == 0: at_1 += 1.0
        if rank_pos < 2: at_2 += 1.0
        if rank_pos < 10: at_10 += 1.0
        if rank_pos < 5:  at_5 += 1.0
        mrr += 1.0/float(rank_pos+1)
        
    final_mrr = mrr/float(total_size)
    acc1 = at_1/float(total_size)
    acc2 = at_2/float(total_size)
    acc5  = at_5/float(total_size)
    acc10 = at_10/float(total_size)

    path,wtype = __return_file_type(workdir,dtype)
    __print_score(path,wtype,acc1,acc2,acc5,acc10,final_mrr)


def score_mono(workdir,ranks,gold_pos,k,dtype):
    """Score the monolingual rank dataset

    :param workdir: the working directory 
    :param ranks: the generated ranks 
    :param gold_pos: the gold positions 
    :param rsize
    """
    total_size = ranks.shape[0]
    at_1  = 0.0
    at_2  = 0.0
    at_10 = 0.0
    at_5  = 0.0
    mrr = 0.0
    
    for i in range(total_size):
        rank_size = ranks[i].shape[0]
        rank_pos = gold_pos[i]
        if rank_pos == 0: at_1 += 1.0
        if rank_pos < 2: at_2 += 1.0
        if rank_pos < 10: at_10 += 1.0
        if rank_pos < 5:  at_5 += 1.0
        mrr += 1.0/float(rank_pos+1)

    ## final scores
    final_mrr = mrr/float(total_size)
    acc1 = at_1/float(total_size)
    acc2 = at_2/float(total_size)
    acc5  = at_5/float(total_size)
    acc10 = at_10/float(total_size)

    ## print the score
    path,wtype = __return_file_type(workdir,dtype)
    __print_score(path,wtype,acc1,acc2,acc5,acc10,final_mrr)

    ## print the ranks
    __print_poly_ranks(workdir,ranks,gold_pos,k=k)
    
def score_poly(workdir,langs,ranks,rmap,gold_pos,k,dtype,exc=False):
    """Score a polyglot language set of ranks 

    :param workdir: the target working directory 
    :param langs: the language identifiers 
    :param ranks: the actual ranks 
    :param rmap: the map of ranked output 
    :param gold_pos: the gold positions in each rank 
    :param dtype: the type of data being evaluated 
    """
    total_size = ranks.shape[0]
    at_1  = 0.0
    at_2  = 0.0
    at_10 = 0.0
    at_5  = 0.0
    mrr = 0.0
    lang_count = defaultdict(int)
    lang_occ  = defaultdict(float)
    
    ## find the number of components per language
    if exc:
        lang_count = {}
        for lang in langs:
            lang_count[lang] = len(rmap)
    else:
        for component in rmap.keys():
            lang = component.split()[0].strip()
            lang_count[lang] += 1

    ## score per language 
    lang_scores = {l:{"1":0.0,"2":0.0,"10":0.0,"5":0.0,"mrr":0.0} for l in lang_count.keys()}

    ## first pass to get global score 
    for i in range(total_size):
        language = langs[i]
        lang_occ[language] += 1.0

        rank_size = ranks[i].shape[0]
        rank_pos = gold_pos[i]

        ## put it as last in the rank if not there
        ## should probably be if rank_pos >= (rank_size -1)
        if rank_pos == (rank_size -1):
            rank_pos = lang_count[language]-1
            
        if rank_pos == 0:
            at_1 += 1.0
            lang_scores[language]["1"] += 1.0
                
        if rank_pos < 2:
            at_2 += 1.0
            lang_scores[language]["2"] += 1.0
            
        if rank_pos < 10:
            at_10 += 1.0
            lang_scores[language]["10"] += 1.0
            
        if rank_pos < 5:
            at_5 += 1.0
            lang_scores[language]["5"] += 1.0
            
        mrr += 1.0/float(rank_pos+1)
        lang_scores[language]["mrr"] += 1.0/float(rank_pos+1)

    ## global scores
    final_mrr = mrr/float(total_size)
    acc1      = at_1/float(total_size)
    acc2      = at_2/float(total_size)
    acc5      = at_5/float(total_size)
    acc10     = at_10/float(total_size)

    path,wtype = __return_file_type(workdir,dtype)
    __print_score(path,wtype,acc1,acc2,acc5,acc10,final_mrr)

    ## print scores for each dataset individually
    for lang,scores in lang_scores.items():
        lang_total = lang_occ[lang]
        final_mrr = scores["mrr"]/lang_total if lang_total > 0 else 0.0
        acc1      = scores["1"]/lang_total if lang_total > 0 else 0.0
        acc2      = scores["2"]/lang_total if lang_total > 0 else 0.0
        acc5      = scores["5"]/lang_total if lang_total > 0 else 0.0
        acc10     = scores["10"]/lang_total if lang_total > 0 else 0.0
        path,wtype = __return_file_type(workdir,dtype)
        __print_score(path,wtype,acc1,acc2,acc5,acc10,final_mrr,descr=lang)

    ## print the ranks
    __print_poly_ranks(workdir,ranks,gold_pos,k=k)

def __backup_ranks(wdir,dtype,old_ranks,new_ranks,gold_info,iteration):
    """Print a backup fot he ranks


    :param wdir: the working directory 
    :param dtype: the type of data being tested 
    :param new_ranks: the new ranks to print 
    :param iteration: the current iteration
    """
    __rank_results_file(wdir)
    file_out = os.path.join(wdir,"rank_results/%s_ranks_%d.txt" % (dtype,iteration))
    num_items = new_ranks.shape[0]
    
    if not os.path.isfile(file_out):
        with codecs.open(file_out,'w',encoding='utf-8') as ranks:
            for i in range(num_items):
                gp = gold_info[i]
                gold_identifier = old_ranks[i][gp]
                reranked = new_ranks[i]
                print >>ranks,"%d\t%d\t%s" %\
                  (i,gold_identifier,' '.join([str(k) for k in reranked[:100]]))
                
def compare_ranks(old_ranks,new_ranks,gold_pos,## rank information
                      dtype, ## the type of data being evaluated
                      iteration,likelihood,
                      wdir,scorer,other_gold):
    """Compare discriminative model score after reranking
    
    :param old_ranks: the original baseline ranks 
    :param new_ranks: the new ranks 
    :param dtype: the data type (e.g., training/test,...) 
    
    :param iteration: the current number of iterations performed 
    :param likelihood: the current data (log) likelihood with model

    :param wdir: the working directory 

    :rtype: A score object
    """
    num_items = gold_pos.shape[0]

    at_1  = 0.0
    at_2  = 0.0
    at_5  = 0.0
    at_10 = 0.0
    mrr   = 0.0

    ## entries in dataset
    util_logger.info('Evaluate new ranks, other_gold=%s' % str(other_gold != {}))

    if not other_gold:

        ## normal evaluation, there's a lot of redundancy with what's below
        ## but I don't want to break the evaluator that works

        for item in range(num_items):
            gp = gold_pos[item]
            gold_identifier = old_ranks[item][gp]
            reranked = new_ranks[item]
            rsize = reranked.shape[0]
        
            ## if gold position is beyond the beam, just use position from oroginal ranking
            if gp > rsize:
                raise ValueError('Gold value larger than rank size!: %d' % gp)
            #     mrr += 1.0/float(gp+1.0)
            #     continue

            ## individual ranks
            for pos in range(rsize):
                rid = reranked[pos]
                ## use old ranks if 
                if rid == -1:
                    #raise ValueError('New ranked value is still -1!')
                    ## something weird here 
                    rid = old_ranks[item][pos]
                    ## shut off for now 
                    #util_logger.info('Rank item is still -1: pos=%d' % pos)
                    
                if (rid == gold_identifier) and (pos == 0): at_1 += 1.0
                if (rid == gold_identifier) and (pos < 2): at_2 += 1.0
                if (rid == gold_identifier) and (pos < 10): at_10 += 1.0
                if (rid == gold_identifier) and (pos < 5): at_5 += 1.0
                if (rid == gold_identifier):mrr += 1.0/float(pos+1.0)

    else:

        for item in range(num_items):
            gp = gold_pos[item]
            gold_identifier = old_ranks[item][gp]
            reranked = new_ranks[item]
            rsize = reranked.shape[0]
            accepted = set([gold_identifier])
            best_index = None

            if other_gold.get(item,None):
                for identifier in other_gold[item]:
                    accepted.add(old_ranks[item][identifier])
            
            ## gold position is larger than rank size 
            if gp > rsize:
                raise ValueError('Gold value larger than rank size!: %d' % gp)

            for pos in range(rsize):
                rid = reranked[pos]
                if rid == -1: rid = old_ranks[item][pos]

                if rid in accepted and best_index == None:
                    best_index = pos

            if best_index == 0: at_1 += 1.0
            if best_index <  2: at_2 += 1.0
            if best_index < 10: at_10 += 1.0
            if best_index < 5:  at_5 += 1.0
            ## assign some large number of not found
            if best_index == None: best_index = 10000
            mrr += 1.0/float(best_index+1.0)

    final_mrr = mrr/float(num_items)
    acc1      = at_1/float(num_items)
    acc2      = at_2/float(num_items)
    acc5      = at_5/float(num_items)
    acc10     = at_10/float(num_items)

    path,wtype = __return_file_type(wdir,dtype)
    msg = "<%s> run after %s iterations or removing feature %s" % (dtype,iteration+1,iteration)
    __print_score(path,wtype,acc1,acc2,acc5,acc10,final_mrr,msg,ll=str(likelihood))

    ## backup ranks
    if 'select' not in dtype: 
        __backup_ranks(wdir,dtype,old_ranks,new_ranks,gold_pos,iteration)
    
    return scorer(acc1,acc10,final_mrr)


def compare_multi_ranks(langs,old_ranks,new_ranks,gold_pos,dtype,iteration,likelihood,wdir,scorer,other_gold):
    """Computes scores for multingual datasets, or datasets containing other sub datasets

    
    """
    num_items = gold_pos.shape[0]

    ## global scores
    at_1  = 0.0
    at_2  = 0.0
    at_5  = 0.0
    at_10 = 0.0
    mrr   = 0.0

    ## language scores
    lang_scores = {}
    lang_totals = {}

    ## make lang directory
    lang_results = os.path.join(wdir,"lang_results")
    ## new language result directory 
    if not os.path.isdir(lang_results):os.mkdir(lang_results)
    util_logger.info('Evaluate new ranks, other_gold=%s' % str(other_gold != {}))
    
    ## datases without other gold 
    if not other_gold:
        
        for item in range(num_items):
            ## language
            language = langs[item]
            if language not in lang_scores:
                lang_scores[language] = {"at_1":0.0,"at_2":0.0,"at_5":0.0,"at_10":0.0,"mrr":0.0}
                lang_totals[language] = 0.0

            lang_totals[language] += 1.0
            
            gp = gold_pos[item]
            gold_identifier = old_ranks[item][gp]
            reranked = new_ranks[item]
            rsize = reranked.shape[0]
            found = False                         

            for pos in range(rsize):
                rid = reranked[pos]
                if rid == -1:
                    rid = old_ranks[item][pos]

                if rid == gold_identifier:
                    found = True 
                    
                ## ordinary scoring
                if (rid == gold_identifier) and (pos == 0):
                    lang_scores[language]["at_1"] += 1.0 
                    at_1 += 1.0
                if (rid == gold_identifier) and (pos < 2):
                    lang_scores[language]["at_2"] += 1.0 
                    at_2 += 1.0
                if (rid == gold_identifier) and (pos < 10):
                    lang_scores[language]["at_10"] += 1.0 
                    at_10 += 1.0
                if (rid == gold_identifier) and (pos < 5):
                    lang_scores[language]["at_5"] += 1.0 
                    at_5 += 1.0
                if (rid == gold_identifier):
                    lang_scores[language]["mrr"] += 1.0 /float(pos+1.0)
                    mrr += 1.0/float(pos+1.0)

            ## when it is not in beam 
            if not found:
                mrr += 1.0/float(gp)
            
    else:

        for item in range(num_items):

            ## language
            language = langs[item]

            if language not in lang_scores:
                lang_scores[language] = {"at_1":0.0,"at_2":0.0,"at_5":0.0,"at_10":0.0,"mrr":0.0}
                lang_totals[language] = 0.0
            lang_totals[language] += 1.0
            gp = gold_pos[item]
            gold_identifier = old_ranks[item][gp]
            reranked = new_ranks[item]
            rsize = reranked.shape[0]
            accepted = set([gold_identifier])
            best_index = None

            if other_gold.get(item,None):
                for identifier in other_gold[item]:
                    accepted.add(old_ranks[item][identifier])

            for pos in range(rsize):
                rid = reranked[pos]
                if rid == -1: rid = old_ranks[item][pos]

                if rid in accepted and best_index == None:
                    best_index = pos

            if best_index == 0:
                lang_scores[language]["at_1"] += 1.0 
                at_1 += 1.0
            if best_index <  2:
                lang_scores[language]["at_2"] += 1.0 
                at_2 += 1.0
            if best_index < 10:
                lang_scores[language]["at_10"] += 1.0 
                at_10 += 1.0
            if best_index < 5:
                lang_scores[language]["at_5"] += 1.0 
                at_5 += 1.0
            ## assign some large number of not found
            if best_index is None:
                best_index = 10000

            lang_scores[language]["mrr"] += 1.0/float(best_index+1.0)
            mrr += 1.0/float(best_index+1.0)

    ## score on overall dataset
    final_mrr = mrr/float(num_items)
    acc1      = at_1/float(num_items)
    acc2      = at_2/float(num_items)
    acc5      = at_5/float(num_items)
    acc10     = at_10/float(num_items)

    path,wtype = __return_file_type(wdir,dtype)
    msg = "<%s> run after %s iterations or removing feature %s" % (dtype,iteration+1,iteration)
    __print_score(path,wtype,acc1,acc2,acc5,acc10,final_mrr,msg,ll=str(likelihood))

    ## backup ranks
    if 'select' not in dtype: 
        __backup_ranks(wdir,dtype,old_ranks,new_ranks,gold_pos,iteration)

    ## individual languages
    lang_file_out = os.path.join(lang_results,"%s_%d.txt" % (dtype,iteration+1))
    printed = False

    ## average accuracy
    aacc1  = 0.0
    aacc2  = 0.0
    aacc5  = 0.0
    aacc10 = 0.0
    amrr   = 0.0

    for (lang,scores) in lang_scores.items():
        lang_total = lang_totals[lang]
        final_mrr = scores["mrr"]/lang_total if lang_total > 0 else 0.0
        acc1      = scores["at_1"]/lang_total if lang_total > 0 else 0.0
        acc2      = scores["at_2"]/lang_total if lang_total > 0 else 0.0
        acc5      = scores["at_5"]/lang_total if lang_total > 0 else 0.0
        acc10     = scores["at_10"]/lang_total if lang_total > 0 else 0.0
        file_out = 'w' if not printed else 'a'
        __print_score(lang_file_out,file_out,acc1,acc2,acc5,acc10,final_mrr,descr=lang)
        printed = True

        ## add to average 
        aacc1  += acc1
        aacc2  += acc2
        aacc5  += acc5
        aacc10 += acc10
        amrr   += final_mrr

    aacc1  = aacc1/float(len(lang_scores)) if aacc1 > 0 else 0.0
    aacc2  = aacc2/float(len(lang_scores)) if aacc2 > 0 else 0.0
    aacc5  = aacc5/float(len(lang_scores)) if aacc5 > 0 else 0.0
    aacc10 = aacc10/float(len(lang_scores)) if aacc10 > 0 else 0.0
    amrr   = amrr/float(len(lang_scores)) if amrr > 0 else 0.0

    ## print to file
    apath,awtype = __return_file_type(wdir,"valid_average")
    amsg = "validation average run after %s iterations" % str(iteration+1)
    __print_score(apath,awtype,aacc1,aacc2,aacc5,aacc10,amrr,amsg)

    ## back up the ranks
    if 'select' not in dtype: 
        __backup_ranks(wdir,dtype,old_ranks,new_ranks,gold_pos,iteration)
    return scorer(aacc1,aacc10,amrr)
