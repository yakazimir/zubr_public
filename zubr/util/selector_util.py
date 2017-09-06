# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Utilities for feature selection 

"""
import os
import re
import codecs
import logging
import gzip
from collections import defaultdict
from zubr.util.optimizer_util import __eval_script as make_script

util_logger = logging.getLogger('zubr.util.selector_util')

def __find_selection_settings(config):
    """Figure out what to call the resulting model 

    :param config: the configuration 
    :rtype: str
    :returns: path name 
    :raises: ValueError 
    """
    if config.test_templates and config.test_individual:
        base = "select_indv_temp"
        name = os.path.join(config.model_dir,base)
        
    elif config.test_templates:
        base = "select_temp"
        name = os.path.join(config.model_dir,base)

    elif config.test_individual:
        base = "select_indv"
        name = os.path.join(config.model_dir,base)
        
    else:
        util_logger.warning('No selection method selected, choosing individual...')
        base = "select_indv"
        name = os.path.join(config.model_dir,base)
        config.test_individual = True

    return (base,name)

def load_selector(config):
    """Load a feature selector

    :param config: the main configuration 
    """
    templates      = __get_templates(config)
    valid_features = __get_valid_path(config)

def restore_config(config):
    """Restore a given configuration

    :param config: the main configuration 
    """
    if not config.model_dir:
        raise ValueError('Must specify the model location!')
    
    config.restore_old(config.model_dir,ignore=[])
    name,model_path = __find_selection_settings(config)

    ## remove previous runs
    vsr = os.path.join(config.model_dir,"VALID_SELECT_RESULTS.txt")
    if os.path.isfile(vsr):
        os.remove(vsr)

    ## remove previous models (if there are here)
    model_name = os.path.join(config.model_dir,model_path)
    if os.path.isfile(model_name):
        os.remove(model_name)


def __find_rank_file(path):
    """Find the rank files (there should be a lot)

    :param path: path to rank result directory
    """
    cand = [os.path.join(path,i) for i in os.listdir(path) if 'valid_ranks_' in i]
    last = ''
    top = 0
    for c in cand:
        rank_info = re.search(r'valid\_ranks\_([0-9]+)\.txt$',c)
        #num = c.split('.')[0].split('_')[-1]
        num = int(rank_info.groups()[0])
        #num = int(num)
        if num >= top: last = c
    wrong = []

    ## find incorrect examples
    with codecs.open(last,encoding='utf-8') as my_ranking:
        for line in my_ranking:
            number,gold,ranks = line.split('\t')
            rank_list = [int(i) for i in ranks.split()]
            gold = int(gold)
            number = int(number)
            if rank_list[0] != gold:
                
                try:
                    gindex = rank_list.index(gold)
                    others = rank_list[:gindex]
                    assert gold not in others
                    wrong.append([number,gold,set(others)])
                except ValueError:
                    pass
    return wrong

def wrong_features(config):
    """Opens validation features and finds incorrect features 

    :param config: the main feature selection configuration 
    :returns: commonly encountered feature in wrong analyses with frequency 
    :rtype: dict
    """
    candidates = set()

    ## check that rank results are available

    rank_results = os.path.join(config.model_dir,"rank_results")
    if not os.path.isdir(rank_results):
        raise ValueError('Cannot find the rank directory: %s' % rank_results)

    ## check that valid features are available
    valid_features = os.path.join(config.model_dir,"valid_features")
    if not os.path.isdir(valid_features):
        raise ValueError('Cannot find the validation features')

    incorrect_list = __find_rank_file(rank_results)
    counts = defaultdict(int)
    frequency = defaultdict(int)

    for (number,gold,others) in incorrect_list:
        feature_file = os.path.join(valid_features,"%d.gz" % number)
        gold_features  = defaultdict(int)
        other_features  = defaultdict(int)
        
        ## read the features 
        with gzip.open(feature_file,'rb') as my_instance:
            gold_found = False
            
            for line in my_instance:
                line = line.strip()
                identifier,feature_list = line.split('\t')
                fvals = [int(i.split("=")[0]) for i in feature_list.split()]
                identifier = int(identifier)
                ## go through the features 
                for feature in fvals:
                    frequency[feature] += 1
                    if identifier == gold:
                        gold_found = True 
                        gold_features[feature] += 1
                    elif identifier in others:
                        other_features[feature] += 1

            #assert gold_found,"Gold features not found %s" % str(number)

            for (feature,count) in other_features.iteritems():
                if feature not in gold_features:
                    counts[feature] += count

    return dict(counts)

def backup_better(config,dumper,removed):
    """Backup the better model 

    :param config: the main configuration 
    :param dumper: the dumper function 
    :param removed: the remove templates 
    """

    name,model_path = __find_selection_settings(config)
    #new_model = os.path.join(config.model_dir,"selected_model")

    new_model = model_path #o#.path.join(config.model_dir,model_path)
    # ## remove existing model 
    if os.path.isfile(new_model+".lz4"): os.remove(new_model+".lz4")
    # ## dump the model

    util_logger.info('Dumping the selected model...')
    dumper(new_model)

    # ## create testing script
    make_script(config.model_dir,"--eval_test",'%s_test' % name,new_model)
    make_script(config.model_dir,"--eval_val",'%s_valid' % name,new_model)
    # make_script(config.model_dir,"--eval_train",'%s_train' % name,new_model)

    # ##
    if removed: 
        removed_info = os.path.join(config.model_dir,"TEMPLATES_REMOVED.txt")
        with codecs.open(removed_info,'w',encoding='utf-8') as my_removed:
            print >>my_removed,"%s" % (' '.join([str(i) for i in removed]))

        ## make a script for loading retraining from scratch
