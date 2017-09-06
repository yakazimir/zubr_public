# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Building the aligner utility 

"""

import os
import time
import logging
import pickle
import codecs
import numpy as np
from collections import defaultdict
from zubr.util.aligner_util import get_tree_data
from zubr.util.hiero import read_hiero_grammar,read_phrase_table
from zubr.Features import TemplateManager

## utlity logger

util_logger = logging.getLogger('zubr.util.alinger_extractor')

__all__ = [
    "find_base",
    "build_extractor",
]

ENC = 'utf-8'

TEMPLATES = {
    ##############################################
    # ## LEXICAL FEATURES AND ALIGNMENT FEATURES #
    ##############################################
    0  : "Model beam position of target (binned)",
    1  : "Number of unigram containments (binned)",
    2  : "Unigram text/component pairs", # might shut off or modify to  
    3  : "Number of unigram matches (binned, zero to five)",
    4  : "Type of unigram matches",
    5  : "Tree positions of match",
    6  : "Number of bigram matches (binned, zero to five)",
    7  : "Number of bigram containments (binned, zero to five)",
    8  : "E bigram is contained in F bigram (binned)",
    9  : "F bigram is contained if E bigram (binned)",
    ## viterbi alignment 
    10 : "Viterbi alignment positions",
    11 : "Tree position of viterbi alignment positions",
    #11 : "All positions in f string are covered? (binary)",
    12 : "All positions in f string are covered given component length?",
    13 : "All tree positions are aligned to",
    14 : "Word and component length comparison (binned)",
    #######################
    # RAW PHRASE FEATURES #
    #######################
    15 : "Phrase instances extracted from training",
    16 : "Number of phrase instances recognized (binned, zero to 5)",
    17 : "Number of matching phrases (binned)",
    18 : "Number of phrase containments (binned)",
    19 : "Length of matching phrases (binned)",
    20 : "Tree position of phrases",
    21 : "Tree position of phrase matches",
    ##########################
    # Hiero Phrase Features  #
    ##########################
    22 : "Hiero rules from hiero grammar (extracted from training)",
    23 : "Number of valid rules recognized (binned)",
    ## removed 24,25,26,27,28
    24 : "Number of hiero rule string matches (binned)",
    25 : "Number of left context matches (binned)",
    26 : "Number of right context matches (binned)",
    27 : "Size of hiero rule string contexts (binned)",
    28 : "Size of hiero rule string contexts given lhs (binned)",
    29 : "Number of hiero rules with NT re-ordering (binned)",
    30 : "Types of NT re-ordering",
    31 : "Hiero rule gap sizes (binned)",
    32 : "Training frequency of rules (binned)",
    ##########################
    # TECH DOCUMENT FEATURES #
    ##########################
    33 : "Unigram pair appears in param description",
    34 : "Number of pairs in descriptions (binned)",
    35 : "Word unigram/abstract category pair types",
    36 : "See also class pair/unigram",
    37 : "See also class/pair unigram match (binned)",## template composition 
    38 : "Tree position of item/description pair",    ## template composition
    #########################
    # COMPOSITION FEATURES  #
    #########################
    39 : "Viterbi alignment and pair in description (binned)",
    40 : "Viterbi alignment, pair in description, tree position (binned)",
    41 : "Pair match and in description (binned)",
    42 : "Pair Match, in description, and tree position",
    43 : "Phrases and see-also classes",
    44 : "Hiero rules and spans in description",
    45 : "Hiero rules and abstract see-also classes",
    46 : "Abstract class and sentence length (binned=3)",
    ## extra lexical features (in first case)
    #########################################
    47 : "ebigram matches f word (binned)",
    48 : "fbigram matched e word (binned)",
    49 : "bigram-word match (binned)",
    50 : "tree position of unigram containments",
    51 : "tree position of bigram matches",
    52 : "contiguous bigram both in description of f word",
    53 : "number of contiguous bigrams both in descriptions (binned)",
    54 : "tree position of contiguous bigrams in description",
    55 : "viterbi alignment and match (binned)",
    ## extra phrase featues
    ###########################################
    56 : "size of english phrase in matched phrases (binned)",
    57 : "size of foreign phrase in matched phrases (binned)",
    58 : "size of english phrase in known phrases (binned)",
    59 : "size of foreign phrase in known phrases (binned)",
    60 : "size of english phrase in overlapping phrases (binned)",
    61 : "size of foreign phrase in overlapping phrases (binned)",
    ## composition features 
    62 : "english phrase + abstract description symbol (> 1)",
    63 : "phrase in descriptions, foreign phrase (> 1)",
    64 : "size of english phrases in description pair",
    65 : "size of foreign phrase in description pair",
    66 : "Foreign phrase and english input length",
    67 : "E-side ends and begins with words in descriptions",
    ## normal phrase stuff
    68 : "Tree position of known phrases",
    69 : "Tree position of matching phrases",
    70 : "Tree position of phrases with english side in descriptions",
    71 : "Tree position of phrases containments",
    72 : "Tree position of phrases with overlapping words",
    73 : "Size of word overlap for overlapping phrases",
    74 : "Length of english input given individual phrases",
    75 : "Tree distance for phrases",
    ## knowledge features
    76 : "Abstract symbols encountered (regardless of part. unigram)",
    77 : "Particular english unigrams in descriptions",
    78 : "Particular foreign descriptions",
    79 : "Tree position of abstract_symbol pair",
    ## extra phrase features
    80 : "Abstract symbols and phrases",
    81 : "Abstract symbols and viterbi alignments",
    82 : "Number of phrase/abstact symbol pairs (binned)",
    ## base features 
    83 : "e word ids",
    84 : "f word ids",
    85 : "e input length",
    86 : "f input length",
    87 : "e input length (binned)",
    88 : "f input length (binned)",
    ### base hiero features
    89 : "english side hiero rules",
    90 : "foreign side hiero rules",
    91 : "type of classes and reordering",
    92 : "number of unknown rules (binned)",
    93 : "abstract classes and english side of hiero rule",
    94 : "abstract classes and foreign side of hiero rule",
    95 : "word overlap in hiero rule (binned)",
    96 : "word overlap in hiero rule and lhs",
    }

def find_base(config):
    """return the path to the base aligner model

    :param config: the main configuration
    :rtype: str
    :raises: ValueError
    """
    ## the base model should be called ``base.model`` 
    path = os.path.join(config.dir,"base.model")
    if not os.path.isfile(path+".gz") and not os.path.isfile(path):
        raise ValueError('Cannot find base aligner model: %s' % path)
    return path

def __class_indices(path):
    """find indices of class items in rank file"""
    cfile = os.path.join(path,"orig_data/rank_list_class.txt")
    if not os.path.isfile(cfile):
        util_logger.warning('No classes indices found...')
        return ({},0)

    cpositions = {}
    total_classes = set()
    with codecs.open(cfile,encoding=ENC) as my_classes:
        for k,line in enumerate(my_classes):
            cpositions[k] = {}
            line = line.strip()
            line_spots = line.split()
            for index in range(len(line_spots)):
                class_id = int(line_spots[index])
                if class_id < 0: continue
                cpositions[k][index] = class_id
                total_classes.add(class_id)
                
    return (cpositions,max(total_classes)+1)

def __abstract_file(path,edict,fdict):
    """return a map of abstract word/symbol pairs
    
    :param path: directory path
    :param edict: english dictionary (to encode words)
    :param fdict: foreign dictionary (to encode words)
    :rtype: dict
    """
    a = os.path.join(path,"orig_data/abstract_categories.txt")
    total = defaultdict(set)
    if not os.path.isfile(a):
        util_logger.warning('No abstract categories found...')
        return (dict(total),0)

    total_count = 0
    with codecs.open(a,encoding=ENC) as abstract_classes:
        for k,line in enumerate(abstract_classes):
            line = line.strip()
            if not line: continue 
            aclass,examples = line.split('\t')
            for ex in examples.split():
                ex = ex.strip().lower()
                total[ex] = k
                total_count = k

    return (dict(total),total_count)

def __description_file(path,edict,fdict):
    """Parse the descroption file

    :param path: directory path
    :param edict: english dictionary (to encode words)
    :param fdict: foreign dictionary (to encode words)
    :rtype: dict    
    """
    d = os.path.join(path,"orig_data/descriptions.txt")
    total = set()
    if not os.path.isfile(d):
        util_logger.warning('No abstract description pairs found...')
        return {}

    with codecs.open(d,encoding=ENC) as descriptions:
        for k,line in enumerate(descriptions):
            line = line.strip()
            symbol,words = line.split('\t')
            symbol = symbol.strip().lower()
            for word in words.split():
                word = word.strip().lower()
                total.add((word,symbol))
                
    return {i:k for k,i in enumerate(total)}


def __feature_templates(settings,config):
    """Determines the feature templates to use 

    :param config: the main configuration 
    :returns: the set of feature templates strings
    :rtype: set
    """
    templates = set(config.atemplates.split('+'))
    settings.has_phrase    = True if "phrase"    in templates else False
    settings.has_knowledge = True if "knowledge" in templates else False
    settings.has_compose   = True if "compose"   in templates else False
    settings.has_hiero     = True if "hiero"     in templates else False

def __overlap(config,e,f):
    """find matching words in aligner vocabulary, ignores some stop words
    
    :param e: english side dictionary/map
    :type e: dict
    :param f: foreign side dictionary/map
    :type f: dict
    :rtype: setn 
    """
    stops = ['a','the','of','an']
    m = set([(e[w],f[w]) for w in e if w in f and w not in stops and len(w) > 3])
    matches = {i:k for k,i in enumerate(m)}
    config.word_matches = matches

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

def __read_trees(config,settings):
    """Read the component tree file for the rank set (required)


    :param config: the main configuration 
    :param settings: the new settings configuration 
    :rtype: None
    """
    try: 
        tree_pos = get_tree_data(config,ex="orig_data",tset="rank")
    except Exception,e:
        util_logger.error(e,exc_info=True)
        raise ValueError('error reading tree file: %s' % e)
    finally:
        settings.tree_pos = tree_pos
        settings.treemax = config.treemax


# def __compute_knowledge_size(settings,temp_sizes):
#     """Estimate the templates sizes of teh knowledge or document features

#     :param settings: the current settings 
#     :param temp_sizes: the templates size object 
#     """
#     settings.num_abstract = len(settings.abstract_classes)
#     position_dict,num_class = __class_indices(settings.dir)
#     settings.num_classes = num_class
#     settings.class_items = position_dict

#     temp_sizes[33] = len(settings.descriptions)
#     temp_sizes[34] = settings.bin_size
#     temp_sizes[35] = settings.num_abstract*settings.elen
#     temp_sizes[36] = settings.elen*settings.num_classes
#     temp_sizes[37] = settings.bin_size
#     temp_sizes[38] = settings.treemax
#     #temp_sizes[38] = settings.bin_size
    
def __compute_lex_size(settings,temp_sizes,flen):
    """Estimate the size of the base lexical features

    :param settings: the current settings 
    :param temp_sizes: the template size dictionary 
    :param elen: the size of english vocabulary
    :rtype: None
    """
    ## compute phrases from table 

    temp_sizes[0]  = 15
    temp_sizes[1]  = 5
    temp_sizes[2]  = settings.product 
    temp_sizes[3]  = settings.bin_size
    temp_sizes[4]  = settings.product #len(settings.word_matches)
    temp_sizes[5]  = settings.treemax
    temp_sizes[6]  = settings.bin_size
    temp_sizes[7]  = settings.bin_size
    temp_sizes[8]  = settings.bin_size
    temp_sizes[9]  = settings.bin_size
    temp_sizes[10] = settings.product
    temp_sizes[11] = settings.treemax
    #temp_sizes[12] = 2
    temp_sizes[12] = 0
    #temp_sizes[13] = 2
    temp_sizes[13] = 0      
    #temp_sizes[14] = flen*10 ## needs to be computed somehow
    ##shut off 
    temp_sizes[14] = 0
    temp_sizes[47] = settings.bin_size
    temp_sizes[48] = settings.bin_size
    temp_sizes[49] = settings.bin_size
    temp_sizes[50] = settings.treemax
    temp_sizes[51] = settings.treemax
    temp_sizes[52] = settings.flen
    temp_sizes[53] = settings.bin_size
    temp_sizes[54] = settings.treemax
    temp_sizes[55] = settings.bin_size
    #temp_sizes[83] = settings.elen
    # shut off
    temp_sizes[83] = 0
    #temp_sizes[84] = settings.flen
    # shut off 
    temp_sizes[84] = 0
    #temp_sizes[85] = settings.amax
    # shut off 
    temp_sizes[85] = 0
    #temp_sizes[86] = settings.amax
    temp_sizes[86] = 0
    #temp_sizes[87] = settings.bin_size
    temp_sizes[87] = 0
    #temp_sizes[88] = settings.bin_size
    # shut off 
    temp_sizes[88] = 0
    
    
def __compute_phrase_size(settings,temp_sizes):
    """Compute phrase features and sets the size
    
    :param config: the overall configuration 
    :param settings: the new settings for extractor 
    :param temp_sizes: the template sizes
    :rtype: None 
    """
    ## compute phrases from phrase table file 
    #settings.phrase_map = read_phrase_table(settings.dir)
    #phrase_map,english_map = read_phrase_table(settings.dir)
    phrase_map,english_map,foreign_map = read_phrase_table(settings.dir)

    settings.phrase_map = phrase_map
    settings.english_map = english_map
    settings.foreign_map = foreign_map
    settings.num_phrases = len(settings.phrase_map)
    settings.num_en_phrases = len(settings.english_map)
    settings.num_fr_phrases = len(settings.english_map)

    temp_sizes[15] = settings.num_phrases
    #temp_sizes[16] = settings.bin_size
    ##shuf off 
    temp_sizes[16] = 0
    temp_sizes[17] = settings.bin_size
    temp_sizes[18] = settings.bin_size
    temp_sizes[19] = settings.bin_size
    #temp_sizes[19] = 0
    #temp_sizes[20] = settings.treemax*settings.bin_size
    temp_sizes[20] = 0
    #temp_sizes[21] = settings.treemax*settings.bin_size
    temp_sizes[21] = 0
    temp_sizes[56] = settings.bin_size
    temp_sizes[57] = settings.bin_size
    #temp_sizes[58] = settings.bin_size
    #shut off 
    temp_sizes[58] = 0
    #temp_sizes[59] = settings.bin_size
    ## shut off 
    temp_sizes[59] = 0
    #temp_sizes[60] = settings.bin_size
    ## shut off 
    temp_sizes[60] = 0
    #temp_sizes[61] = settings.bin_size
    ## shuf off 
    temp_sizes[61] = 0
    temp_sizes[68] = settings.treemax
    temp_sizes[69] = settings.treemax
    temp_sizes[70] = settings.treemax
    temp_sizes[71] = settings.treemax
    temp_sizes[72] = settings.treemax
    #temp_sizes[66] = settings.bin_size
    temp_sizes[66] = 0
    temp_sizes[67] = settings.bin_size
    temp_sizes[73] = settings.bin_size
    #temp_sizes[74] = (settings.num_fr_phrases*5)
    temp_sizes[74] = 0
    #temp_sizes[75] = settings.treemax
    temp_sizes[75] = 0

def __compute_hiero_size(settings,temp_sizes):
    """Compute the size and position of hiero rules 

    :param settings: the settings to add things to 
    """
    hiero,glue,lhs,es,fs = read_hiero_grammar(settings.hierogrammar,settings.gluegrammar)
    settings.hiero    = hiero
    settings.glue     = glue
    settings.lhs_glue = lhs
    settings.num_hiero = len(settings.hiero)
    settings.num_glue  = len(settings.lhs_glue)
    settings.hes        = es
    settings.hfs        = fs
    settings.numhe      = len(settings.hes)
    settings.numhf      = len(settings.hfs)
    ## update temp sizes
    temp_sizes[22] = settings.num_hiero
    temp_sizes[23] = settings.bin_size
    #temp_sizes[24] = settings.bin_size
    temp_sizes[24] = 0
    temp_sizes[25] = settings.bin_size
    temp_sizes[25] = 0
    #temp_sizes[26] = settings.bin_size
    temp_sizes[26] = 0
    #temp_sizes[27] = settings.bin_size
    temp_sizes[27] = 0
    #temp_sizes[28] = settings.bin_size
    temp_sizes[28] = 0
    temp_sizes[29] = settings.bin_size
    #temp_sizes[30] = settings.num_glue*settings.bin_size
    #temp_sizes[31] = settings.bin_size
    temp_sizes[31] = 0
    #temp_sizes[32] = settings.bin_size
    temp_sizes[32] = 0
    temp_sizes[30] = settings.num_glue
    temp_sizes[89] = settings.numhe
    temp_sizes[90] = settings.numhf
    temp_sizes[92] = settings.bin_size
    #temp_sizes[95] = settings.bin_size
    temp_sizes[95] = 0
    #temp_sizes[96] = settings.num_glue
    temp_sizes[96] = 0

def __compute_knowledge_size(settings,temp_sizes):
    """Estimate the templates sizes of teh knowledge or document features

    :param settings: the current settings 
    :param temp_sizes: the templates size object 
    """
    #settings.num_abstract = len(settings.abstract_classes)
    position_dict,num_class = __class_indices(settings.dir)
    settings.num_classes = num_class
    settings.class_items = position_dict
    settings.num_descriptions = len(settings.descriptions)

    temp_sizes[33] = len(settings.descriptions)
    temp_sizes[34] = settings.bin_size
    temp_sizes[35] = settings.num_abstract*settings.elen
    temp_sizes[36] = settings.elen*settings.num_classes
    temp_sizes[37] = settings.bin_size
    temp_sizes[38] = settings.treemax
    temp_sizes[76] = settings.num_abstract
    if settings.num_descriptions: 
        temp_sizes[77] = settings.elen
        temp_sizes[78] = settings.flen
    temp_sizes[79] = settings.treemax
    temp_sizes[81] = settings.elen*settings.num_abstract

    # settings.num_abstract = len(settings.abstract_classes)
    # position_dict,num_class = __class_indices(settings.dir)
    # settings.num_classes = num_class
    # settings.class_items = position_dict
    # settings.num_descriptions = len(settings.descriptions)

    # temp_sizes[33] = len(settings.descriptions)
    # temp_sizes[34] = settings.bin_size
    # temp_sizes[35] = settings.num_abstract
    # temp_sizes[36] = settings.elen*settings.num_classes
    # temp_sizes[37] = settings.bin_size
    # temp_sizes[38] = settings.bin_size    

def __compute_compose_size(settings,temp_sizes):
    """Computes the sizes associated with composition features


    :param settings: the main settings 
    :param temp_sizes: the individual template sizes 
    """
    if settings.num_descriptions: 
        temp_sizes[39] = settings.bin_size
    if settings.num_descriptions: 
        #temp_sizes[40] = settings.bin_size
        temp_sizes[40] = 0
    if settings.num_descriptions: 
        temp_sizes[41] = settings.bin_size
    if settings.num_descriptions: 
        temp_sizes[42] = settings.treemax
    if settings.num_phrases and settings.num_classes:
        #temp_sizes[43] = settings.num_phrases*settings.num_classes
        temp_sizes[43] = settings.num_en_phrases*settings.num_classes
    if settings.num_hiero:
        temp_sizes[44] = 0 ## note sure how to do this
    if settings.num_classes and settings.num_hiero:
        temp_sizes[45] = settings.num_hiero*settings.num_classes
    if settings.num_classes:
        #temp_sizes[46] = settings.num_classes*10
        temp_sizes[46] = 0
    if settings.num_en_phrases:
        temp_sizes[62] = settings.num_en_phrases
        #temp_sizes[64] = settings.bin_size
        temp_sizes[64] = 0
    if settings.num_fr_phrases:
        temp_sizes[63] = settings.num_fr_phrases
        temp_sizes[65] = settings.bin_size
    if settings.num_classes and settings.num_en_phrases:
        temp_sizes[80] = settings.num_classes*settings.num_en_phrases
        temp_sizes[82] = settings.bin_size
    if settings.num_classes and settings.num_hiero:
        temp_sizes[91] = settings.num_classes
    if settings.numhe and settings.num_classes:
        temp_sizes[93] = (settings.numhe*settings.num_classes)
    if settings.numhf and settings.num_classes:
        temp_sizes[94] = (settings.numhf*settings.num_classes)
    

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

def __old_model(settings,temp_sizes):
    """Revert back to an earlier model"""
    temp_sizes[61] = settings.bin_size
    temp_sizes[88] = settings.bin_size
    temp_sizes[64] = settings.bin_size
    temp_sizes[66] = settings.bin_size
    temp_sizes[83] = settings.elen
    temp_sizes[84] = settings.flen
    temp_sizes[85] = 50
    temp_sizes[86] = 50
    temp_sizes[87] = settings.bin_size
    temp_sizes[88] = settings.bin_size
    temp_sizes[24] = settings.bin_size
    temp_sizes[26] = settings.bin_size
    temp_sizes[27] = settings.bin_size
    temp_sizes[28] = settings.bin_size
    temp_sizes[31] = settings.bin_size
    temp_sizes[32] = settings.bin_size
    temp_sizes[19] = settings.bin_size
    temp_sizes[16] = settings.bin_size
    temp_sizes[14] = settings.flen*10
    temp_sizes[59] = settings.bin_size
    temp_sizes[12] = 2
    temp_sizes[13] = 2
    temp_sizes[74] = (settings.num_fr_phrases*5)
    temp_sizes[75] = settings.treemax
    temp_sizes[96] = settings.num_glue
    temp_sizes[40] = settings.bin_size
    temp_sizes[46] = settings.num_classes*10
    temp_sizes[25] = settings.bin_size
    temp_sizes[20] = 15
    temp_sizes[21] = 15
    temp_sizes[95] = settings.bin_size
    temp_sizes[58] = settings.bin_size
    temp_sizes[16] = settings.bin_size
    temp_sizes[60] = settings.bin_size
    
def build_extractor(base_model,config,settings):
    """Build the aligner extractor and target_settings configuration

    :param ex_config: the experiment and aligner configuration
    :param target_settings: the configuration to build 
    :returns: the target_settings object with new values 
    :rtype: zubr.util.config.ConfigAttrs
    """

    util_logger.info('starting the aligner feature extraction...')
    start_time = time.time()
    
    ### template sizes (should assign zero to all items in template map)
    temp_sizes = TemplateManager(TEMPLATES)

    ## lexicons
    elex = base_model.elex
    flex = base_model.flex
    
    ## setup templates and check for missing files 
    templates = __feature_templates(settings,config)

    ### update settings for new target_settings
    settings.dir  = config.dir
    settings.elen = base_model.elen
    settings.flen = base_model.flen
    settings.amax = config.amax
    settings.hierogrammar = config.hierogrammar
    settings.gluegrammar  = config.gluegrammar
    settings.lang = config.lang if config.lang else 'en'
    settings.heuristic  = config.aheuristic
    settings.store_feat = config.store_feat

    ## logging 
    util_logger.info('Extractor language: %s' % settings.lang)

    ## beam size
    settings.beam = config.beam

    __open_rankset(config.dir,settings)                 ## ranks 
    __overlap(settings,base_model.elex,base_model.flex) ## overlapping words

    ## decision: the aligner should include a tree file
    __read_trees(config,settings)

    ##################################################################
    #                                                                #
    # Each feature template occupies a span in the weight vector, so #
    # the size of each template needs to be determined               #
    ##################################################################

    settings.bin_size = 5
    settings.product  = settings.elen*settings.flen

    ## standard lex rule sizes
    __compute_lex_size(settings,temp_sizes,settings.flen)

    ### PHRASE FEATURES
    if settings.has_phrase:
        __compute_phrase_size(settings,temp_sizes)

    if settings.has_hiero:
        __compute_hiero_size(settings,temp_sizes)
    else:
        settings.glue  = {}
        settings.hiero = {}

    if settings.has_knowledge:
        ## descriptions and abstract class files 
        #settings.abstract_classes = __abstract_file(settings.dir,elex,flex)
        abstract_classes,num_a = __abstract_file(settings.dir,elex,flex)
        settings.abstract_classes = abstract_classes
        settings.num_abstract = num_a
        settings.descriptions     = __description_file(settings.dir,elex,flex)
        __compute_knowledge_size(settings,temp_sizes)
    else:
        settings.class_items  = {}
        settings.descriptions = {}
        settings.abstract_classes = {}
        settings.num_abstract = 0
        settings.num_classes = 0

    if settings.has_compose:
        __compute_compose_size(settings,temp_sizes)
    else:
        pass

    ## revert back to earlier model (e.g., for c)
    if config.old_model:
        settings.old_model = True
        util_logger.info('Building the old feature model!')
        __old_model(settings,temp_sizes)

    ## blocks
    #fblist = config.temp_blocks.split(")
    if config.temp_blocks:
        for feat_temp in config.temp_blocks.split('+'):
            temp_sizes[int(feat_temp)] = 0

    ## important 
    temp_sizes.compute_starts()
    temp_sizes.print_description(settings.dir)
    settings.num_features = temp_sizes.num_features
    settings.tempmanager = temp_sizes.starts

    ## store features?
    __store_feature_setup(settings.store_feat,settings.dir)

    #settings.tempmanager = temp_sizes
    util_logger.info('Finished building in %s seconds' % str(time.time()-start_time))
    util_logger.info('Extractor has %d features' % settings.num_features)
