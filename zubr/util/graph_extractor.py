import os
import sys
import codecs
import shutil
import time 
import logging
import subprocess
import pickle 
import numpy as np
from zubr.Features import TemplateManager

from zubr.Phrases import (
    HieroPhraseTable,
    SimplePhraseTable,
    SparseDictWordPairs,
    DescriptionWordPairs
)    

from zubr.FeatureComponents import (
    WordPhraseComponents,
    PolyComponents,
    MonoComponents,
    KnowledgeComponents,
    StorageComponents
)

from zubr.util.aligner_util import get_tree_data
from zubr.Dataset import RankStorage

__all__ = [
    "build_extractor",
    "run_concurrent",
]

util_logger = logging.getLogger('zubr.util.graph_extractor')

## feature templates 
TEMPLATES = {
    ######################################
    # ## LEXICAL AND ALIGNMENT FEATURES  #
    ######################################
    0   : "Model beam position",
    1   : "Number of unigram containments",
    2   : "Unigram text/component pairs",
    3   : "Number of unigram matches (binned, zero to five)",
    4   : "Type of unigram matches",
    5   : "Tree positions of match",
    6   : "Number of bigram matches (binned, zero to five)",
    7   : "Number of bigram containments (binned, zero to five)",
    8   : "E bigram is contained in F bigram (binned)",
    9   : "F bigram is contained in E bigram (binned)",
    10  : "E word ids",
    11  : "F word ids",
    12  : "Ebigram matches f word (binned)",
    13  : "Fbigram matches e word (binned)",
    14  : "Viterbi alignment position",
    15  : "Tree position of viterbi alignment positions",
    16  : "Viterbi aligned and match (binned)",
    17  : "Overlap and tree position",
    18  : "Bigram match and tree position",
    19  : "bigram containment and tree position",
    20  : "length of bigram match",
    21  : "length of bigram containment",
    ####################
    # PHRASE FEATURES  #
    ####################
    22  : "Phrase instances extracted from training",
    23  : "Number of phrase instances recognized (binned)",
    24  : "Number of matching phrases (binned)",
    25  : "Number of phrase containments (binned)",
    26  : "Tree position of phrase containments",
    27  : "Size of the english phrases in matched phrases (binned)",
    28  : "Size of foreign phrase in matched phrases (binned)",
    29  : "Known English side phrases",
    30  : "Known Foreign side phrases",
    31  : "Tree position of matching phrases",
    #########################
    # Hiero Phrase Features #
    #########################
    32 : "Known hiero rules from hiero grammar",
    33 : "Number of valid rules recognized (binned)",
    34 : "English side hiero rules",
    35 : "Foreign side hiero rules",
    ######################
    # KNOWLEDGE FEATURES #
    ######################
    36 : "Number of items in description pair (binned)",
    37 : "English side in description pair",
    38 : "Foreign side in description pair",
    39 : "Occurence of an abstract class",
    40 : "E Unigram with abstract class",
    41 : "Tree position of description pair",
    ############################
    # COMPOSITION OF FEATURES  #
    ############################
    42 : "See also classes and E phrase rules",
    43 : "See also classes and F hiero rules",
    44 : "Match and in description (binned)",
    45 : "Match and type of class (binned)",
    46 : "See also classes and full hiero rules",
    47 : "Phrase begin and ends in descriptions (binned)",
    48 : "Matches and abstract category types",
    49 : "In description and in viterbi alignment (binned)",
    50 : "In description and tree position",
    ##########################
    # TRANSLATION/PARAPHRASE #
    ##########################
    51 : "Size of english phrases in phrase overlap",
    52 : "Size of english phrases in phrase overlap",
    53 : "Size of known english phrases",
    54 : "Size of known foreign phrases",
    55 : "LHS rules covered by hiero search",
    56 : "Reordering and lhs side types",
    57 : "Number of reordered hiero rules (binned)",
    58 : "E hiero rules and abstract classes",
    ###############################
    # language specific features  #
    ###############################
    59 : "Language and beam position",
    60 : "Language and number of unigram containments",
    61 : "Language and number of unigram matches",
    62 : "Language and E word ids",
    63 : "Language and F word ids",
    64 : "Language and unigram pairs",
    65 : "Language and bigram matches",
    66 : "Language and bigram containments",
    67 : "Language and type of matches",
    68 : "Language and matchers",
    69 : "Language and contains description pair",
    70 : "Language and class type",
    71 : "The language identifier",
    72 : "Language and phrase entries",
    73 : "Language and hiero entries",
    74 : "Phrase match and language",
    75 : "Phrase containment and language",
    76 : "Language and e side of hiero phrases",
    77 : "Language and f side of hiero phrases",
    78 : "Language and hiero reordering",
    79 : "identifier of description pair",
    80 : "language and description pair identifier",
}

def __check_dependencies(config,settings):
    """Checks dependencies and transfers settings between global config and local config 
    
    :param config: the global configuration 
    :param settings: the local extractor config
    """
    ## copy over settings 
    settings.dir        = config.dir
    settings.store_feat = config.store_feat
    settings.heuristic  = config.aheuristic
    settings.lang       = config.lang if config.lang else 'en'
    settings.amax       = config.amax
    #settings.beam       = config.beam
    settings.beam       = config.k
    settings.bin_size   = 5
    settings.glue       = config.glue
    settings.ispolyglot = config.ispolyglot
    settings.concur_extract = config.concur_extract
    settings.num_extract  = config.num_extract

    ## information about the model
    settings.timeout      = config.timeout 
    settings.extractor    = config.extractor
    settings.decoder_type = config.decoder_type
    settings.modeltype    = config.modeltype
    settings.eval_val     = config.eval_val
    settings.exc_model    = config.exc_model
    ## set the offset to zero 
    settings.offset       = 0

    ## log language info
    util_logger.info('Extractor language: %s' % str(settings.lang))
    ## possible dependencies
    config.phrase_file = os.path.join(config.dir,"phrase_table.txt")
    config.descr_file  = os.path.join(config.dir,"descriptions.txt")
    settings.phrase_file = config.phrase_file
    settings.descr_file = config.descr_file
    
    ## check for description file
    
    ## check for hiero grammar
    if not config.hierogrammar:
        config.hierogrammar = os.path.join(config.dir,"hiero_rules.txt")
        settings.hierogrammar = config.hierogrammar
        
    if not config.glue:
        config.glue = os.path.join(config.dir,"grammar.txt")
        settings.glue = config.glue

    ## check that hiero grammar dependencies exist 
    if config.hierogrammar and not os.path.isfile(config.hierogrammar):
        util_logger.warning('Hiero rules not found: %s' % config.hierogrammar)
        config.hierogrammar = None
        settings.hierogrammar = None 
        
    if config.glue and not os.path.isfile(config.glue):
        util_logger.warning('No glue grammar rules found: %s' % config.glue)
        config.hierogrammar = None
        config.glue = None
        settings.hierogrammar = None
        settings.glue = None 
        
    if not os.path.isfile(config.phrase_file):
        util_logger.warning('No phrase file found: %s' % config.phrase_file)
        config.phrase_file = None
        settings.phrase_file = None 
        
    if not os.path.isfile(config.descr_file):
        util_logger.warning('No description file found: %s' % config.descr_file)
        config.descr_file = None
        settings.descr_file = None

def __compute_lex_size(settings,temp_sizes):
    """Compute the size for the class of lexical features 

    :param settings: the extractor configuration 
    :param temp_sizes: the object holidng the template sizes (to be modified here)
    """
    temp_sizes[0] = 15
    temp_sizes[1] = settings.bin_size
    temp_sizes[2] = settings.product
    temp_sizes[3] = settings.bin_size
    temp_sizes[4] = settings.flen
    temp_sizes[5] = 20
    temp_sizes[6] = settings.bin_size
    temp_sizes[7] = settings.bin_size
    temp_sizes[8] = settings.bin_size
    temp_sizes[9] = settings.bin_size
    temp_sizes[10] = settings.elen
    temp_sizes[11] = settings.flen
    temp_sizes[12] = settings.bin_size
    temp_sizes[13] = settings.bin_size
    temp_sizes[14] = settings.product
    #temp_sizes[15] = 20
    temp_sizes[15] = 0
    temp_sizes[16] = settings.bin_size
    temp_sizes[17] = 20
    temp_sizes[18] = 20
    temp_sizes[19] = 20
    #temp_sizes[20] = settings.bin_size
    #temp_sizes[21] = settings.bin_size
    temp_sizes[20] = 0
    temp_sizes[21] = 0

def __compute_hiero_size(container,settings,temp_sizes):
    """Compute the size for the hiero rules 

    :param settings: the extractor configuration 
    :param temp_sizes: the size of each template 
    """
    num_lhs = max(container.lhs_lookup.values())
        
    temp_sizes[32] = container.num_phrases
    temp_sizes[33] = settings.bin_size
    temp_sizes[34] = container.elen
    temp_sizes[35] = container.flen
    temp_sizes[55] = num_lhs
    temp_sizes[56] = num_lhs
    temp_sizes[57] = settings.bin_size

    settings.num_hiero  = container.num_phrases
    settings.num_ehiero = container.elen
    settings.num_fhiero = container.flen

def __compute_phrase_size(container,settings,temp_sizes):
    """Compute the size for the hiero rules 

    :param size: the size of the phrase table 
    :param settings: the extractor configuration 
    :param temp_sizes: the size of each template 
    """
    temp_sizes[22] = container.num_phrases
    temp_sizes[23] = settings.bin_size
    temp_sizes[24] = settings.bin_size
    temp_sizes[25] = settings.bin_size
    temp_sizes[26] = 20
    temp_sizes[27] = settings.bin_size
    temp_sizes[28] = settings.bin_size
    temp_sizes[29] = container.elen
    temp_sizes[30] = container.flen
    temp_sizes[31] = 20
    temp_sizes[51] = settings.bin_size
    temp_sizes[52] = settings.bin_size
    temp_sizes[53] = settings.bin_size
    temp_sizes[54] = settings.bin_size

    settings.num_phrases = container.num_phrases
    settings.num_ephrase = container.elen
    settings.num_fphrase = container.flen

def __compute_compose_size(settings,temp_sizes):
    """Compute the size and number of composition features 

    :param settings: the extractor configuration 
    :param temp_sizes: the template or feture type counts
    """
    num_phrases = 0 if not settings.num_phrases else settings.num_phrases
    num_hiero   = 0 if not settings.num_hiero   else settings.num_hiero
    num_ephrases = 0 if not settings.num_phrases  else settings.num_ephrase
    num_fphrases = 0 if not settings.num_phrases  else settings.num_fphrase
    num_ehiero = 0 if not settings.num_hiero  else settings.num_ehiero
    num_fhiero = 0 if not settings.num_hiero  else settings.num_fhiero
    num_trees = 20
    num_classes = 0 if not settings.num_classes else settings.num_classes

    temp_sizes[42] = num_classes*num_ephrases
    #temp_sizes[43] = num_classes*num_ehiero
    temp_sizes[43] = 0
    temp_sizes[44] = settings.bin_size
    temp_sizes[45] = num_classes*settings.bin_size
    #temp_sizes[46] = num_classes*num_hiero
    temp_sizes[46] = 0
    temp_sizes[47] = settings.bin_size
    temp_sizes[48] = num_classes
    temp_sizes[49] = settings.bin_size
    #temp_sizes[50] = num_trees
    temp_sizes[58] = num_ehiero*num_classes
    
def __feature_templates(settings,config):
    """Determines the feature templates to use 

    :param config: the main configuration 
    :returns: the set of feature templates strings
    :rtype: set
    """
    templates = set(config.atemplates.split('+'))
    settings.has_phrase     = True if "phrase"     in templates else False
    settings.has_knowledge  = True if "knowledge"  in templates else False
    settings.has_compose    = True if "compose"    in templates else False
    settings.has_hiero      = True if "hiero"      in templates else False
    settings.has_paraphrase = True if "paraphrase" in templates else False

def __read_classes(wdir):
    """Read the class rank list file

    :param wdir: the working directory path 
    :rtype: dict 
    """
    rfile = os.path.join(wdir,"rank_list_class.txt")
    if not os.path.isfile(rfile):
        util_logger.warning('No class file found: %s' % rfile)
        return {}

    entries = {}
    classes = set() 
    ## go through the file 
    with codecs.open(rfile) as my_classes:
        for k,line in enumerate(my_classes):
            line = line.strip()
            entries[k] = {}
            for j,sequence in enumerate(line.split()):
                sequence = int(sequence)
                if sequence == -1: continue
                entries[k][j] = sequence
                classes.add(sequence)

    try: 
        return (entries,max(classes))
    except:
        return {}

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
                

def __setup_ranks(config,settings):
    """Setup the rank items for use in the discriminative reranker 
    
    :param settings: the global extractor settings
    :param config: the global experiment configuration 
    """
    ## load the rank list
    data_ranks = os.path.join(config.dir,"ranks.data")
    archive = np.load(data_ranks)
    rank_list = archive["arr_0"]
    rank_vals = archive["arr_1"]
                     
    ## tree data for the rank list
    tree_pos = get_tree_data(config,tset='rank')
    ## abstract classes associated with rank items 
    #classes,max_class = __read_classes(config.dir)
    v = __read_classes(config.dir)
    if v != {}:
        classes,max_class = v
    else:
        classes =  {}
        max_class = 0
    
    settings.num_classes = max_class

    if rank_list.shape[0] != tree_pos.shape[0]:
        raise ValueError('Size mismatch between rank list and tree positions!')
    if classes and len(classes) != rank_list.shape[0]:
        raise ValueError('Size mismatch between classes and rank_list')

    util_logger.info('Building rank instance...')
    if config.ispolyglot:
        ### extract language information
        return PolyComponents(rank_list,rank_vals,tree_pos,classes)
    return MonoComponents(rank_list,rank_vals,tree_pos,classes)

def __compute_knowledge_size(settings,temp_sizes,psize):
    """Compute the knowledge size

    """
    num_classes = 0 if not settings.num_classes else settings.num_classes
    temp_sizes[36] = settings.bin_size
    temp_sizes[37] = settings.elen
    temp_sizes[38] = settings.flen
    temp_sizes[39] = num_classes
    temp_sizes[40] = settings.elen*num_classes
    temp_sizes[41] = 20
    temp_sizes[79] = psize
    temp_sizes[80] = psize 

def __create_storage(config,settings):
    """Load a storage item with pre-computed ranks 
    
    NOTE: since the graph decoder takes so long to run, you should
    precompute the ranks ahead of time and dump as a backup file .

    It the train ranks are not found, this function will 
    raise a ValueError 
    
    :param config: the main configuration 
    :raises: ValueError
    """
    train_ranksp = os.path.join(config.dir,"train_storage")
    valid_ranksp = os.path.join(config.dir,"valid_storage")
    test_ranksp  = os.path.join(config.dir,"test_storage")
    
    if not os.path.isfile(train_ranksp+".npz"):
        util_logger.error('Cannot find the train ranks, exiting (path=%s)' % train_ranksp)
        exit('Cannot find the train ranks! %s' % train_ranks)

    train_ranks = RankStorage.load_backup(config,name='train')
    ## make sure that beam is not the same size as the ranks, since ranks always contain gold 
    #if config.beam >= train_ranks.rank_size: 
    settings.beam = (train_ranks.rank_size-10)

    util_logger.info('Beam size=%d' % settings.beam)
    query_ranks = RankStorage.load_empty(0,0)

    ## validation ranks 
    if not os.path.isfile(valid_ranksp+".npz"):
        valid_ranks = RankStorage.load_empty(0,0)
        util_logger.warning('Did not find valid ranks, is this intentional? (path=%s)' % valid_ranksp)
    else:
        valid_ranks = RankStorage.load_backup(config,name='valid')

    ## test ranks
    if not os.path.isfile(test_ranksp+".npz"):
        test_ranks = RankStorage.load_empty(0,0)
        util_logger.warning('Did not find test ranks, is this intentional? (path=%s)' % test_ranksp)
    else:
        test_ranks = RankStorage.load_backup(config,name='test')
    return StorageComponents(train_ranks,valid_ranks,test_ranks,query_ranks)

def __block_features(config,temp_sizes):
    """Block specific feature templates if specified 

    :param config: the main configuration 
    :param temp_sizes: the temp sizes (which might be set to zero)
    :rtype: None 
    """
    if not config.feat_blocks:
        util_logger.info('No blocked features!')
        return
    util_logger.info('Blocking features: %s' % config.feat_blocks)
    feat = [int(i) for i in config.feat_blocks.split("+")]
    for f in feat: temp_sizes[f] = 0

def __language_features(settings,temp_sizes):
    """Language specific features 

    """
    if settings.num_langs == 0 or settings.num_langs is None: return
    num_langs = settings.num_langs
    num_classes = 0 if not settings.num_classes else settings.num_classes
    num_phrases = 0 if not settings.num_phrases else settings.num_phrases
    num_hiero  = 0 if not settings.num_hiero else settings.num_hiero
    ehiero = 0 if not settings.num_ehiero else settings.num_ehiero
    fhiero = 0 if not settings.num_fhiero else settings.num_fhiero
    
    temp_sizes[59] = num_langs*15
    temp_sizes[60] = num_langs*settings.bin_size
    temp_sizes[61] = num_langs*settings.bin_size
    temp_sizes[62] = num_langs*settings.elen
    temp_sizes[63] = num_langs*settings.flen
    temp_sizes[64] = num_langs*settings.product
    temp_sizes[65] = num_langs 
    temp_sizes[66] = num_langs
    temp_sizes[67] = num_langs*settings.flen
    temp_sizes[68] = num_langs
    temp_sizes[69] = num_langs
    temp_sizes[70] = num_langs*num_classes
    temp_sizes[71] = num_langs
    temp_sizes[72] = num_langs*num_phrases
    temp_sizes[73] = num_langs*num_hiero
    temp_sizes[76] = num_langs*ehiero
    temp_sizes[77] = num_langs*fhiero
    temp_sizes[78] = num_langs

    if num_phrases > 0:
        temp_sizes[74] = num_langs
        temp_sizes[75] = num_langs
        
def build_extractor(config,settings,graph):
    """Initialize and build the graph feature extractor

    :param config: the overall configuration 
    :param settings: the new settings object 
    :param graph: the underlying graph 
    """
    util_logger.info('Starting to build the graph feature extractor...')
    stime = time.time()

    ## template manager
    temp_sizes = TemplateManager(TEMPLATES)
    templates  = __feature_templates(settings,config)

    ## vocabulary size
    settings.elen    = graph.elen
    settings.flen    = graph.flen

    ## check settings
    __check_dependencies(config,settings)

    ## LEXICAL TEMPLATES
    pairs = SparseDictWordPairs.from_ranks(config)
    settings.product = pairs.size
    __compute_lex_size(settings,temp_sizes)

    ## RANK SETS 
    ####
    try:
        util_logger.info('Creating rank instance...')
        ranks = __setup_ranks(config,settings)
        settings.num_langs = ranks.num_langs
        #util_logger.info('Number of languages: %d' % settings.num_langs)
    except Exception,e:
        util_logger.error(e,exc_info=True)

    ## load the storage items

    util_logger.info('Loading the rank storage components...')
    storage = __create_storage(config,settings)
            
    #####################################
    ## phrases features
    
    if settings.has_phrase and settings.phrase_file:
        util_logger.info('Computing phrase related features...')

        ## phrase object 
        phrase_rules = SimplePhraseTable.from_config(config)
        __compute_phrase_size(phrase_rules,settings,temp_sizes)

    else:
        ## create empty phrase thingy
        phrase_rules = SimplePhraseTable.create_empty()
        util_logger.warning('No phrase rules found, created empty container..')

    ####################################
    ## hiero features
    
    if settings.has_hiero and settings.hierogrammar and settings.glue:
        util_logger.info('Computing hiero related features...')

        ## hiero phrase object 
        hiero_rules = HieroPhraseTable.from_config(config)
        __compute_hiero_size(hiero_rules,settings,temp_sizes)
    else:
        hiero_rules = HieroPhraseTable.create_empty()
        util_logger.warning('No hiero rules found, created empty container...')

    ####################################
    ## knowledge features
    description = DescriptionWordPairs.from_ranks(config)
    util_logger.info('Loaded description pairs, with size=%d' % description.size)
    
    if settings.has_knowledge:
        __compute_knowledge_size(settings,temp_sizes,description.size)

    ## language features
    __language_features(settings,temp_sizes)
    
    ## paraphrase features (empty for now)

    ## composition
    if settings.has_compose:
        __compute_compose_size(settings,temp_sizes)

    #if config.feat_blocks:
    __block_features(config,temp_sizes)
    
    ## compute the template sizes 
    temp_sizes.compute_starts()
    temp_sizes.print_description(settings.dir)
    settings.num_features = temp_sizes.num_features
    settings.tempmanager = temp_sizes.starts

    ## WORD COMPONENT INFO
    util_logger.info('Creating word phrase component instance...')
    wc = WordPhraseComponents(pairs,phrase_rules,hiero_rules)

    util_logger.info('Creating knowledge instance...')
    kc = KnowledgeComponents(description)

    ## setup the feature backup dirs
    __store_feature_setup(settings.store_feat,settings.dir)

    ## FINISH  
    util_logger.info('Set up feature extractor in %s seconds..' % (time.time()-stime))
    return (wc,ranks,kc,storage)

def __make_script(config,wdir,dtype,offset):
    """Make the run script for asynchronous extractor job 

    :param config: the main configuration
    :param dtype: the type of data to extract from 
    """
    log_file = os.path.join(wdir,"%s_log.log" % dtype)
    script_name  = os.path.join(wdir,"%s_job.sh" % dtype)
    script = "./run_zubr feature_extract --epath %s --pipeline_backup --elog %s --extractor_job %s --doffset %d" %\
      (wdir,log_file,dtype,offset)

    ## print the script to file 
    with open(script_name,'w') as my_script:
        print >>my_script,script

    ## give it the right permissions
    subprocess.call(['chmod','755',script_name])
    util_logger.info('Built script: %s' % script_name)
    

### scripts for concurrently running feature extractor

def __join_results(config,dsize,dtype):
    """Join the resulting feature representations printed to file 

    :param config: the overall configuration 
    :param dtype: the data type 
    """
    util_logger.info('Now trying to join the results ')
    
    total = 0
    num_jobs = config.num_extract
    jobs = os.path.join(config.dir,"extractor_jobs")
    splits = dsize/num_jobs
    remainder = dsize % num_jobs 
    main_features = os.path.join(config.dir,"%s_features" % dtype)
    if not os.path.isdir(main_features): os.mkdir(main_features)

    for i in range(num_jobs):
        util_logger.info('Joining job: %d' % i)
        job_loc = os.path.join(jobs,"job_%d" % i)
        features = os.path.join(job_loc,"%s_features" % dtype)
        ## check that the directory exists 
        if not os.path.isdir(features): raise ValueError('Directory not found: %s' % features)

        ## go through
        lsize = splits if i-1 != num_jobs else splits + remainder
        for k in range(lsize):
            ffile = os.path.join(features,"%s.gz" % str(total+k))
            if not os.path.isfile(ffile):
                #raise ValueError('Feature item not found: %s/%s' % (job_loc,ffile))
                util_logger.warning('Feature item not found: %s, will extract later...' % ffile)
                continue 
            
            ## copy over the file
            shutil.copy(ffile,main_features)

        total += lsize
    util_logger.info('Rewrote %d feature files' % total)

def __run_scripts(config,dtype):
    """Runs the individual job scripts 

    :param config: the global configuration 
    :param dtype: the type of data
    """
    num_jobs = config.num_extract
    # start time 
    stime = time.time()
    jobs = os.path.join(config.dir,"extractor_jobs")
    job_list = []

    ## initiate the jobs 
    for i in range(num_jobs):
        job_loc = os.path.join(jobs,"job_%s" % i)
        job_log = os.path.join(job_loc,"job_err.log")
        job_out = os.path.join(job_loc,"job_stdout.log")
        run_script = os.path.join(job_loc,"%s_job.sh" % dtype)

        with open(job_log,'w') as my_job:
            with open(job_out,'w') as my_job2:
                j = subprocess.Popen(run_script,stderr=my_job,stdout=my_job2,shell=True)
                util_logger.info('Running job %d' % i)
                job_list.append(j)

    ## wait for them to finish
    for k,job in enumerate(job_list):
        ## wait until it finishes
        job.wait()
        util_logger.info('Finished job %d' % k)

    ### log the time
    util_logger.info('Finished jobs in %s seconds' % str(time.time()-stime))
    
def __make_dependencies(config,data,dtype):
    """Make the job directories and split up the datasets
    
    :param config: the extractor configuration object 
    :param data: the data to extract for and split up
    """
    num_jobs = config.num_extract
    stime = time.time()

    ## overall job directory 
    jobs = os.path.join(config.dir,"extractor_jobs")
    if not os.path.isdir(jobs): os.mkdir(jobs)

    ## models to be copied
    ftoe      = os.path.join(config.dir,"ftoe")
    etof      = os.path.join(config.dir,"etof")
    graph     = os.path.join(config.dir,"graph")
    decoder   = os.path.join(config.dir,"graph_decoder")
    extractor = os.path.join(config.dir,"graph_extractor")
    phrases   = os.path.join(config.dir,"phrase_data")
    start_size = 0

    
    ## job directories to create 
    for i in range(num_jobs):
        job_loc = os.path.join(jobs,"job_%d" % i)
        if not os.path.isdir(job_loc): os.mkdir(job_loc)

        ## etof model 
        copy1 = subprocess.Popen('cp -r %s %s' % (etof,job_loc),shell=True)
        copy1.wait()
        ## ftoe model 
        copy2 = subprocess.Popen('cp -r %s %s' % (ftoe,job_loc),shell=True)
        copy2.wait()
        ## graph
        copy3 = subprocess.Popen('cp -r %s %s' % (graph,job_loc),shell=True)
        copy3.wait()
        ## decoder 
        copy4 = subprocess.Popen('cp -r %s %s' % (decoder,job_loc),shell=True)
        copy4.wait()
        ## finally, the extractor
        copy5 = subprocess.Popen('cp -r %s %s' % (extractor,job_loc),shell=True)
        copy5.wait()
        ##
        copy6 = subprocess.Popen('cp -r %s %s' % (phrases,job_loc),shell=True)
        copy6.wait()

        ## copy the config
        config.print_to_yaml(job_loc)
        
        ## make directories
        feature_dir = os.path.join(job_loc,"%s_features" % dtype)
        if not os.path.isdir(feature_dir): os.mkdir(feature_dir)

        ## make run script
        offset = start_size
        __make_script(config,job_loc,dtype,offset)
        start_size += data.size/num_jobs
                    
    dtime = time.time()
    data.split_dataset(config.dir,dtype,'extractor_jobs',num_jobs)
    util_logger.info('Split up the dataset in %s seconds' % str(time.time()-stime))
    

def run_concurrent(config,dataset,dtype):
    """Run the extractor asynchronously using a number of processed

    :param config: the extractor configuration 
    :param dataset: the target dataset to extract for 
    :param dtype: the type of data extracting for 
    """
    util_logger.info('Setting up the job infrastructure, num_jobs=%d' % config.num_extract)

    try: 
        ## make the directories, copies over all model files, creates scripts
        __make_dependencies(config,dataset,dtype)
        ## runs the scripts
        __run_scripts(config,dtype)
        ## join the results
        __join_results(config,dataset.size,dtype)
    except Exception,e:
        util_logger.info(e,exc_info=True)

    ## make the script for the individual jobs
