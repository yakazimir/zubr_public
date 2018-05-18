import re
import os
import time
import sys
import codecs
from collections import defaultdict
import numpy as np
import shutil
import logging 
import subprocess
import time

__all__ = [
    "load_graph",
    "language_blocks",
    "setup_jobs",
    "generate_scripts",
    "rerun_script",
    #"setup_neural_jobs",
]
    
util_logger = logging.getLogger('zubr.util.decoder_util')


def rerun_script(config):
    """Write a script template for rerunning the decoder under various conditions

    :param config: the global experiment configuration
    :rtype: None
    """
    script = os.path.join(config.dir,"test_model.sh")
    jlog = "" if not config.jlog else jlog
    spec = "" if not config.spec_lang else "--spec_lang"

    with codecs.open(script,'w',encoding='utf-8') as my_script:
        print >>my_script,"./run_zubr neural --decoder %s --k %d --eval_set %s --mode decoder --graph_beam %s --seed %s --mem %s --rfile %s --atraining %s %s --graph %s --model %s --timeout %f --num_jobs %d --from_neural *PATH_TO_MODEL*" %\
          (config.decoder,
               config.k,
               config.eval_set,
               config.graph_beam,
               config.seed,
               config.mem,
               config.rfile,
               config.atraining,
               spec,
               config.graph,
               config.model,
               config.timeout,
               config.num_jobs
          )
        
    subprocess.call(['chmod','755',script])

    ## reteraining script
    new_script = os.path.join(config.dir,"retrain.sh")

    with codecs.open(new_script,'w',encoding='utf-8') as my_script:
        print >>my_script,"./run_zubr neural --decoder %s --k %d --eval_set %s --mode decoder --graph_beam %s --seed %s --mem %s --rfile %s --atraining %s %s --graph %s --model %s --timeout %f --num_jobs %d --from_neural *PATH_TO_MODEL* --more_train --epochs %d --wdir %s --name %s" %\
          (config.decoder,
               config.k,
               config.eval_set,
               config.graph_beam,
               config.seed,
               config.mem,
               config.rfile,
               config.atraining,
               spec,
               config.graph,
               config.model,
               config.timeout,
               config.num_jobs,
               config.epochs,
               config.wdir,
               config.name
          )

    subprocess.call(['chmod','755',new_script])

    ## add new logging directory
        
def __read_rules(path):
    """Read the line of the graph file 

    :param path: the path to the graph file 
    """
    edge_list   = []
    span_list   = []
    nodes       = set()
    edge_labels = []
    langs       = {}

    ## keep track of current start node
    curr_node = None
    curr_start = None
    start_ends = defaultdict(set)

    ## lookup for edges to words 
    word_map = {}
    
    with codecs.open(path,encoding='utf-8') as my_rule:
        for k,line in enumerate(my_rule):
            entry = line.strip()

            ## skip over empty lines and those with python style comments 
            if not entry or re.search(r'^\#',entry): continue
            ## skip over end nodes 
            elif len(entry.split('\t')) == 1: continue

            try: 
                start,end,word,_ = entry.split('\t')
            except ValueError,e:
                raise ValueError('Error reading graph line: %s' % entry)

            ## update nodes
            word = word.strip().lower()
            start = int(start); end = int(end)
            edge_list.append([start,end])
            edge_labels.append(word)

            ## is a language node?
            if re.search(r'\<\!.+\!\>',word):
                langs[word] = (start,end)
            
            if (start,end) in word_map:
                print (start,end)
            word_map[(start,end)] = word
            
            if curr_node is None:
                curr_node = start
                curr_start = k

            ## next node 
            elif curr_node != start:
                span_list.append([curr_start,k-1])
                curr_node = start
                curr_start = k

            ## add nodes 
            nodes.add(start)
            nodes.add(end)

    ## k-1 as opposed to k (last line is end node label 
    span_list.append([curr_start,k-1])

    assert len(span_list) == len(nodes)-1,"Graph parser error: span list"
    assert len(word_map) == len(edge_list), "Word map not correct: %d/%d" %\
      (len(word_map),len(edge_list))
    assert len(edge_labels) == len(edge_list),"Error computing edge labels"
    edge_list_numpy = np.array([x[-1] for x in edge_list],dtype=np.int32)
    span_list_numpy = np.array(span_list,dtype=np.int32)
    return (edge_list_numpy,span_list_numpy,len(nodes),word_map,edge_labels,langs)

def __encode_map(wmap,labels,lexicon):
    """Encode the words in the wmap with symbol ids

    :param wmap: the target word map 
    :param lexicon: the symbol table
    """
    #encoded_map = {}
    oov = {}
    encoded_labels = []
    encoded_map = {}

    ## oov edges 
    for (pair,word) in wmap.iteritems():
        word = word.strip().lower()
        word_id = lexicon.get(word,-1)
        encoded_map[pair] = word_id
        if word_id == -1:
            oov[pair] = word

    ###
    for word in labels:
        encoded_labels.append(lexicon.get(word,-1))

    elabels = np.array(encoded_labels,dtype=np.int32)
    return (elabels,oov)
        
def load_graph(config,lexicon,poly=False):
    """Load a given directed graph for a decoder 

    :param config: the main configuration
    :param lexicon: the translation/graph lexicon
    :param poly: build for polyglot models? 
    """
    path = config.graph

    if not os.path.isfile(path):
        raise ValueError('Cannot find the graph file!: %s' % path)

    ## read the rule entries
    edges,spans,size,wmap,labels,langs = __read_rules(path)

    ## word encodings 
    elabels,oov = __encode_map(wmap,labels,lexicon)
        
    if poly:
        return (edges,spans,size,elabels,wmap,langs)
    return (edges,spans,size,elabels,wmap)

def language_blocks(config,lang_map):
    """Block certain paths/languages from being generated 

    :param config: the configuration 
    :param lang_map: the edge positions of the different languages 
    :rtype: set 
    """
    block = [] if not config.lang_blocks else config.lang_blocks.split("+")
    if '_blocked' in config.atraining:
        ## a hack for now 
        block = ["php_ru","php_de","php_es","php_fr","php_tr","php_ja","python_ja"]
    
    if not block:
        return set()
    block_edges = set()
    for item in block:
        identifier = "<!%s!>" % item
        if identifier not in lang_map: continue
        block_edges.add(lang_map[identifier])

    util_logger.info('blocking: %s' % ','.join(block))
    return block_edges


## concurrent utilities

def __copy_deps(job_dir,config):
    """Copy over the depencies to a job directory

    :param job_dir: the particular job directory 
    :param config: the main configuration 
    """
    ## the rank file 
    copy1 = subprocess.Popen('cp %s %s' % (config.rfile,job_dir),shell=True)
    copy1.wait()

    ## model depencies

    ## etof translation model
    etof = os.path.join(config.dir,"etof")
    copy2 = subprocess.Popen('cp -r %s %s' % (etof,job_dir),shell=True)
    copy2.wait()

    ## ftoe
    ftoe = os.path.join(config.dir,"ftoe")
    copy3 = subprocess.Popen('cp -r %s %s' % (ftoe,job_dir),shell=True)
    copy3.wait()

    ## graph
    graph = os.path.join(config.dir,"graph")
    copy4 = subprocess.Popen('cp -r %s %s' % (graph,job_dir),shell=True)
    copy4.wait()

    ## graph_decoder
    graph_decoder = os.path.join(config.dir,"graph_decoder")
    copy5 = subprocess.Popen('cp -r %s %s' % (graph_decoder,job_dir),shell=True)
    copy5.wait()

    ## phrases
    phrases = os.path.join(config.dir,"phrase_data")
    copy6 = subprocess.Popen('cp -r %s %s' % (phrases,job_dir),shell=True)
    copy6.wait()

def __copy_neural_deps(job_dir,config):
    """Copy the dependencies for the neural network models 
    
    :param job_dir: the target directory (where the job runs)
    :param config: the configuration 
    :rtype: None 
    """
    ## the rank file 
    copy1 = subprocess.Popen('cp %s %s' % (config.rfile,job_dir),shell=True)
    copy1.wait()

    ## model dependenceis

    ## graph
    graph = os.path.join(config.dir,"graph")
    copy2 = subprocess.Popen('cp -r %s %s' % (graph,job_dir),shell=True)
    copy2.wait()

    ## graph_decoder
    graph_decoder = os.path.join(config.dir,"neural_decoder")
    copy3 = subprocess.Popen('cp -r %s %s' % (graph_decoder,job_dir),shell=True)
    copy3.wait()

    ## neural model
    graph_model = os.path.join(config.dir,"neural_model")
    copy3 = subprocess.Popen('cp -r %s %s' % (graph_model,job_dir),shell=True)
    copy3.wait()

def __copy_data(first,data,size,path,end):
    """Copy slices of the data to each job directory

    """
    copied = subprocess.Popen("head -%d %s | tail -%d > %s.%s" % (first,data,size,path,end),shell=True)
    copied.wait()

def __make_script(job_dir,dtype,name,k,spec,config):
    """Make the job script that will get called

    :param job_dir: the target job directory
    :param dtype: the decoder type
    :param name: the name of the data
    :param k: the size parameters
    """
    ## print the script
    apath = os.path.join(job_dir,name)
    jlog  = os.path.join(job_dir,"job.log")
    spec = "" if not spec else "--spec_lang"
    
    script = "./run_zubr graphdecoder --decoder_type %s --run_model --atraining %s --jlog %s --k %d %s --eval_set %s --amax 80 --modeltype %s" %\
      (dtype,apath,jlog,k,spec,config.eval_set,config.modeltype)

    # print to file 
    script_path = os.path.join(job_dir,"job.sh")
    with open(script_path,'w') as my_script:
        print >>my_script,script

    ## give it the right permissions so that it can be called
    subprocess.call(['chmod','755',script_path])


def __make_neural_script(job_dir,name,config):
    """Make the script for running neural network job

    :param job_dir: the target job directory
    :param config: the global configuration 
    :rtype: None 
    """
    apath = os.path.join(job_dir,name)
    jlog = os.path.join(job_dir,'job.log')
    rfile = os.path.join(job_dir,"rank_list.txt")
    spec = "" if not config.spec_lang else "--spec_lang"
    trace = "" if not config.trace else "--trace"
    copy_m = "" if not config.copy_mechanism else "--copy_mechanism"
    lex_m = "" if not config.lex_model else "--lex_model"

    script = "./run_zubr neural --decoder %s --jlog %s --k %d %s --eval_set %s --amax %s --atraining %s --mode decoder --model_loc %s --graph_beam %d --seed %d --mem %d --rfile %s %s %s %s" %\
      (config.decoder,
           jlog,
           config.k,
           spec,
           config.eval_set,
           config.amax,
           apath,
           job_dir,
           config.graph_beam,
           config.seed,
           config.mem,
           rfile,
           trace,
           copy_m,
           lex_m
           )

    script_path = os.path.join(job_dir,"job.sh")
    with open(script_path,'w') as my_script:
        print >>my_script,script

    ## give it the right permissions so that it can be called
    subprocess.call(['chmod','755',script_path])

def __make_directories_reg(config,suffix,dtype):
    """Make the directory structure for the different jobs

    :param config: the configuration 
    :param suffix: the suffix of the target data to copy
    :param model: a pointer to the model to copy
    :param dtype: the type of target decoder 
    :raises: ValueError
    """
    if config.num_jobs <= 1: raise ValueError('Jobs must be more than 1!')
    num_jobs = config.num_jobs

    jobs_dir = os.path.join(config.dir,"jobs")
    ## create a new jobs directory if this one is used
    if os.path.isdir(jobs_dir):
        job_time = datetime.datetime.fromtimestamp(ts).strftime('jobs_%Y-%m-%d-%H:%M:%S')
        jobs_dir = os.path.join(config.dir,job_time)

    ## update the config 
    config.jobs_dir = jobs_dir
    
    ## try to make the new directory
    if not os.path.isdir(jobs_dir): os.mkdir(jobs_dir)

    ## find the length of the target file to decode
    edata = config.atraining+"%s.e" % suffix
    fdata = config.atraining+"%s.f" % suffix
    if suffix == '':
        edata = config.atraining+"_bow.e"
        fdata = config.atraining+"_bow.f"
        util_logger.info('Decoding train, using bow data')

    ## information about the data 
    data_len    = sum([1 for i in open(edata)])
    size_splits = data_len/num_jobs
    remainder   = data_len % num_jobs
    last        = size_splits
    name        = os.path.basename(config.atraining)
    total_data  = 0

    ## language file (if polyglot dataset) 
    language_file = config.atraining+"%s.language" % suffix
    has_lang = os.path.isfile(language_file)

    for i in range(num_jobs):
        job_dir = os.path.join(jobs_dir,"job_%d" % i)
        ## make directory if it doesnt exist
        if not os.path.isdir(job_dir): os.mkdir(job_dir)
        ndata = os.path.join(job_dir,name+suffix)

        ## copy the model dependencies
        if not 'neural' in dtype:  __copy_deps(job_dir,config)
        else: __copy_neural_deps(job_dir,config)

        ## slice the data
        if i != (num_jobs - 1):
            __copy_data(last,edata,size_splits,ndata,"e")
            __copy_data(last,fdata,size_splits,ndata,"f")

            if has_lang:
                new_lang = os.path.join(job_dir,"%s%s" % (name,suffix)) 
                __copy_data(last,language_file,size_splits,new_lang,"language")

            total_data += size_splits
            last += size_splits
        else:
            __copy_data(data_len,edata,size_splits+remainder,ndata,"e")
            __copy_data(data_len,fdata,size_splits+remainder,ndata,"f")

            ## copy language file (if exists)
            if has_lang:
                new_lang = os.path.join(job_dir,"%s%s" % (name,suffix))
                __copy_data(data_len,language_file,size_splits+remainder,new_lang,"language")

            total_data += (size_splits+remainder)
        ## build the run script
        if not 'neural' in dtype: 
            __make_script(job_dir,dtype,name,config.k,config.spec_lang,config)
        else:
            __make_neural_script(job_dir,name,config)

    util_logger.info('split up %d data points, data_len=%d' % (total_data,data_len))
    return data_len

def __run_jobs(wdir,job_dir):
    """Execute the actual jobs that were set up

    :param wdir: the working directory 
    :rtype: None 
    """
    #job_dir = os.path.join(wdir,"jobs")

    ## keep track of runing jobs 
    running = []
    stime = time.time()

    ## go through the jobs 
    for k,job in enumerate(os.listdir(job_dir)):
        if 'job_' not in job: continue ## possibly other directories
        jscript = os.path.join(job_dir,os.path.join(job,"job.sh"))
        full_path = os.path.join(job_dir,"%s" % job)
        job_log = os.path.join(full_path,"job_err.log")
        job_out = os.path.join(full_path,"job_stdout.log")

        with open(job_log,'w') as my_job:
            with open(job_out,'w') as my_job2:
                j = subprocess.Popen(jscript,stderr=my_job,stdout=my_job2,shell=True)
                util_logger.info('Running job %d' % k)
                running.append(j)

    ## wait until these finish
    for k,job in enumerate(running):
        job.wait()
        util_logger.info('Finished job %d' % k)

    util_logger.info('Finished jobs in %s seconds' % str(time.time()-stime))

def __join_results(wdir,num_jobs,jdir):
    """Glue the results together into one file

    :param wdir: the working directory 
    :param num_jobs: the number of total jobs to join
    :rtype: None
    """
    #jdir = os.path.join(wdir,"jobs")
    joined_ranks = os.path.join(wdir,"merged_ranks.txt")
    rank_items = [os.path.join(jdir,"job_%d/ranks.txt" % i) for i in range(num_jobs)]
    assert len(rank_items) == num_jobs,"not all ranks created!"

    joined_log = os.path.join(wdir,"joining.log")
    with open(joined_log,'w') as jlog:
        joined = subprocess.Popen("cat %s | cut -f 2-3 | nl -v 0 > %s" % (' '.join(rank_items),joined_ranks),
                                    shell=True,
                                    stderr=jlog,
                                    stdout=jlog,
                                    )
        joined.wait()
    #os.system("cat %s | cut -f 2-3 | nl -v 0 > %s" % (' '.join(rank_items),joined_ranks))

def __clear_jobs(wdir):
    """Elimitate the extra data created by the jobs

    :param wdir: the working directory 
    """
    jdir = os.path.join(wdir,"jobs")
    for job in os.listdir(job_dir):
        full_path = os.path.join(job_dir,"%s" % job)
        shutil.rmtree(full_path)


def setup_jobs(config,jtype='reg'):
    """Sets up the various infrastructure for running concurrent jobs

    :param config: the global configuration instance
    """

    dtype = ''
    ## target decoder type
    if config.decoder_type == "con_wordgraph":
        dtype = 'wordgraph'
    elif config.decoder_type == "con_polyglot":
        dtype = 'polyglot'
    elif config.decoder == "con_sp":
        dtype = 'neural_wordgraph'
    elif config.decoder == "con_poly":
        dtype = 'neural_polyglot'
    if not dtype:
        raise ValueError('Unknown target decoder: %s' % dtype)

    ## type of data to decode
    suffix = ''
    if config.eval_set == "valid":
        suffix = '_val'
    elif config.eval_set == "test":
        suffix = '_test'
    elif config.eval_set == "train":
        suffix = ''
    else:
        raise ValueError('Unknown set to decode: %s' % config.eval_set)

    ## make the data 
    try: 
        data_len = __make_directories_reg(config,suffix,dtype)
    except Exception,e:
        util_logger.error('Error building job data!')
        util_logger.error(e,exc_info=True)

    ## run the jobs
    try:
        __run_jobs(config.dir,config.jobs_dir)
    except Exception,e:
        util_logger.error('Error executing jobs!')
        util_logger.error(e,exc_info=True)

    ## glue results together
    try:
        __join_results(config.dir,config.num_jobs,config.jobs_dir)
    except Exception,e:
        util_logger.info('Error glueing the results!')
        util_logger.error(e,exc_info=True)

    ## clean the jobs data
    # try:
    #     __clear_jobs(config.dir)
    # except Exception,e:
    #     util_logger.info('Error cleaning the jobs data')
    #     util_logger.error(e,exc_info=True)
                
    ## return data and rank size 
    rsize = sum([1 for i in open(config.rfile)])
    return (data_len,rsize)

def generate_scripts(config):
    """Generate run scripts for later using the backed up decoder models 

    :param config: the global experimental configuration 
    :rtype: None 
    """
    dtype = ''
    ## target decoder type
    if config.decoder_type == "con_wordgraph":
        dtype = 'wordgraph'
    elif config.decoder_type == "con_polyglot":
        dtype = 'polyglot'
    elif config.decoder == "con_sp":
        dtype = 'neural_wordgraph'
    elif config.decoder == "con_poly":
        dtype = 'neural_polyglot'
    if not dtype:
        raise ValueError('Unknown target decoder: %s' % dtype)

    ### validation jobs
    suffix = '_val'
                

# def setup_neural_jobs(config,jtype='reg'):
#     """Sets up the infrastructure for running concurrent neural jobs 

#     :param config: the global configuration 
#     :type config: zubr.util.config.ConfigAttrs 
#     """
#     dtype = ''
#     ## target decoder type
#     if config.decoder_type == "con_sp":
#         dtype = 'wordgraph'
#     elif config.decoder_type == "con_poly":
#         dtype = 'polyglot'
#     if not dtype:
#         raise ValueError('Unknown target decoder: %s' % dtype)

#     ## type of data to decode
#     suffix = ''
#     if config.eval_set == "valid":
#         suffix = '_val'
#     elif config.eval_set == "test":
#         suffix = '_test'
#     elif config.eval_set == "train":
#         suffix = ''
#     else:
#         raise ValueError('Unknown set to decode: %s' % config.eval_set)

#     ## make the data 
#     try: 
#         data_len = __make_directories_reg(config,suffix,dtype)
#     except Exception,e:
#         util_logger.error('Error building job data!')
#         util_logger.error(e,exc_info=True)

