import time
import os
import sys
import logging
import codecs
import pickle
from zubr.Features import TemplateManager
import numpy as np

__all__ = [
    "build_extractor"
]

TEMPLATES = {
    0 : "Word component pairs",
    1 : "english words",
    2 : "foreign words",
}

util_logger = logging.getLogger('zubr.util.bow_extractor')

ENC = 'utf-8'

def __open_lex(config,settings,temp_sizes,individual):
    """Open lexical symbol tables

    :param config: the main configuration
    :param settings: the new settings 
    :rtype: None
    """
    lex = os.path.join(config.dir,'alignment/lex.p')
    if not os.path.isfile(lex):
        raise ValueError('Cannot find lexicon: %s' % lex)

    with open(lex) as mylex: model_lex = pickle.load(mylex)

    ## back up the symbol tables 
    fdict,edict = model_lex
    settings.fdict = fdict
    settings.edict = edict
    settings.elen   = len(settings.edict)
    settings.flen  = len(settings.fdict)

    ## template update
    temp_sizes[0] = settings.elen*settings.flen

    if individual: 
        temp_sizes[1] = settings.elen
        temp_sizes[2] = settings.flen
    else:
        temp_sizes[1] = 0
        temp_sizes[2] = 0

def __open_rankset(work_path,settings):
    """Open the rank dataset

    :param work_path: the working directory
    :rtype: None
    """
    rank_set = os.path.join(work_path,"ranks.data")
    if not os.path.isfile(rank_set):
        raise ValueError('Cannot find rank set: %s' % rank_set)
    with open(rank_set) as my_ranks:
        data = np.load(my_ranks)
        settings.ranks = data["arr_0"]
        settings.rank_vals = data["arr_1"]
        settings.beam = settings.ranks.shape[0]
        
def __store_feature_setup(store_feat,wdir):
    """Builds a directory in the case that features will be stored 

    :param store_feat: bool indicating whether or not to store features 
    :param wdir: the working experiment directory
    :rtype: None 
    """
    if store_feat:
        util_logger.info('Building offline feature directories...')
        
        train_f = os.path.join(wdir,'train_features')
        valid_f = os.path.join(wdir,'valid_features')
        test_f  = os.path.join(wdir,'test_features')
        if not os.path.exists(train_f):
            os.makedirs(train_f)
        if not os.path.exists(valid_f):
            os.makedirs(valid_f)
        if not os.path.exists(test_f):
            os.makedirs(test_f)        
        
def build_extractor(config,settings):
    """Main function for building a bag of words extractor


    :param config: the main configuration
    :param settings: the new configuration to setup up
    """
    util_logger.info('starting the bow feature extraction...')
    start_time = time.time()
    templates = [i.strip() for i in config.templates.split("+")]
    pair = True
    individual = True if "indv" in templates else False

    temp_sizes = TemplateManager(TEMPLATES)

    ## find lexicon and sizes
    __open_lex(config,settings,temp_sizes,individual)

    ## copy over settings 
    settings.dir  = config.dir
    settings.amax = config.amax

    ## setup the rank information
    __open_rankset(config.dir,settings)

    ## binned matches

    temp_sizes.compute_starts()
    temp_sizes.print_description(settings.dir)
    settings.num_features = temp_sizes.num_features
    settings.tempmanager = temp_sizes.starts
    settings.ignore_stops = config.ignore_stops
    settings.lang = config.lang
    settings.match_feat = config.match_feat
    settings.store_feat = config.store_feat

    __store_feature_setup(settings.store_feat,settings.dir)
    util_logger.info('Finished building in %s seconds' % str(time.time()-start_time))
    util_logger.info('Extractor has %d features' % settings.num_features)
