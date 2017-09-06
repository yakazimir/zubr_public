# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""
import time
import logging
import os
import subprocess
from zubr.Extractor import Extractor
from zubr.FeatureExtractor import ExtractorClass
from zubr.Dataset import Dataset,EMPTY_RANK

PATH = os.path.abspath(os.path.dirname(__file__))
ZUBR_PATH = os.path.abspath(os.path.join(PATH,"../../"))

util_logger = logging.getLogger('zubr.util.optimizer_util')

def make_eval_script(config):
    """After training model create script that can evaluate on test

    :param config: the training configuration 
    """
    pass

def find_extractor(config):
    """Find a feature extractor in working directory

    :param config: main configuration
    """
    util_logger.info('Finding and loading the feature extractor...')

    ## the old backup protocol 
    if not config.pipeline_backup: 
        extractor_loc = os.path.join(config.dir,"extractor.p")
        if not os.path.isfile(extractor_loc+".gz"):
            util_logger.fatal('Working directory not specified!')
            raise ValueError('Cannot find the extractor model: %s' % extractor_loc)
        return Extractor.load(extractor_loc)

    ## new pipeline loader
    eclass    = ExtractorClass(config.extractor)
    extractor = eclass.load_backup(config)
    return extractor

def __check_workdir(config):
    """Check if the working directory is specified

    :param config: the optimizer and experiment configuration
    """
    if not config.dir:
        raise ValueError('No working directory specified, cannot find data...')

def __find_valid(config,main_type):
    """Find the validation data for use during optiizer training
    
    -- assumes that data is in working directory specified as config.dir, 
    and named ``valid.data``

    :param config: the main configuration 
    :param main_type: the main type of data otherwise
    """
    vpath = os.path.join(config.dir,"valid.data")
    if 'test' in main_type: return None

    if config.eval_val and not os.path.isfile(vpath+".gz"):
        raise ValueError('Validation data not found!')

    if config.eval_val:
        return Dataset.load(vpath)
    return EMPTY_RANK

def __find_main(config,name):
    """Find the main data for the optimizer

    -- assumes that all data is in working directory config.dir
    -- it should end with .data

    :param config: the experiment configuration 
    :param name: the prefix name of the dataset type (e.g., train,test)
    :raises: ValueError 
    :returns: The data instances
    :rtype: zubr.Dataset.Data
    """
    dpath = os.path.join(config.dir,"%s.data" % name)
    if not os.path.isfile(dpath+'.gz'):
        raise ValueError('Cannot find the target data: %s' % dpath)

    util_logger.info('Found main data: %s' % dpath)
    return Dataset.load(dpath)

def find_data(config,dtype):
    """Load data for training or testing 

    :param config: the main configuration 
    :param dtype: the type of data to open
    """
    util_logger.info('finding the target data: %s' % dtype)
    __check_workdir(config)
    if dtype == 'valid':
        return __find_valid(config,dtype)
    elif dtype == 'train-test':
        return __find_main(config,'train')
    elif dtype == 'test':
        return __find_main(config,dtype)
    main  = __find_main(config,dtype)
    valid = __find_valid(config,dtype)
    if main and valid:
        return (main,valid)
    return main

def build_optimizer(config,settings):
    """Constract settings for optimizer

    :param config: the overall configuration 
    :param settings: an empty configuration to create 
    :param extractor: the feature extractor
    :returns: the extractor
    """
    util_logger.info('Starting to build the optimizer settings...')
    
    if not config.miters:
        raise ValueError('number of epochs number set, --miters')

    settings.epochs      = config.miters
    settings.shuffle     = config.shuffle
    settings.reg         = config.rlambda
    settings.lrate1      = config.lrate1
    settings.lrate2      = config.lrate2
    settings.dir         = config.dir
    settings.eval_val    = config.eval_val
    settings.eval_train  = config.eval_train
    settings.hyper_param = config.hyper_type
    settings.testset     = config.testset
    settings.rlambda     = config.rlambda
                

def restore_config(config,wdir):
    """Restores an old optimizer configuration
    
    :param config: the rebuilt configuration 
    :param model_loc: the location of the optimizer model 
    :rtype: None 
    """
    config.restore_old(wdir,ignore=["eval_val","eval_test",
                                        "eval_train","restore"])

def backup_current(config,iteration,dumper):
    """Backups up the current model and changes model_loc point in config

    :param config: the current optimizer config 
    :param iteration: the current iteration 
    :param dumper: the dumper function 
    """
    stime = time.time()
    ## remove the previous model
    #if config.model_loc: os.remove(config.model_loc+".gz")
    if config.model_loc: os.remove(config.model_loc+".lz4")
    #new model location 
    mp = os.path.join(config.dir,"model_%s" % str(iteration+1))
    dumper(mp)

    util_logger.info('Backup up model after <%s> iterations in %s seconds...' %\
                         (str(iteration+1),str(time.time()-stime)))

    config.model_loc = mp


def __eval_script(wdir,ttype,name,model_loc):
    """Generate an eval script for testing the trained model 

    :param wdir: the working directory and place to put the script
    :param ttype: the type of evaluation 
    """
    full = "cd %s\n./run_zubr optimizer --restore %s %s --model_loc %s" %\
      (ZUBR_PATH,wdir,ttype,model_loc)
    script_path = os.path.join(wdir,"eval_%s.sh" % name)

    with open(script_path,'w') as my_script:
        print >>my_script,full

    subprocess.call(['chmod','755',script_path])

def reload_model(config,loader):
    """Reload a optimization model from example

    :param config: the main configuration/pointer to model
    :param loader: the class loader 
    :raises: ValueError
    """
    #if not config.model_loc or not os.path.isfile(config.model_loc+".gz"):
    if not config.model_loc or not os.path.isfile(config.model_loc+".lz4"):
        raise ValueError('Cannot find model: %s' % config.model_loc)

    util_logger.info('Loading model %s (might take a few seconds)...' % config.model_loc)
    start_time = time.time()
    m =  loader(config.model_loc)
    util_logger.info('Rebuilt model in %s seconds!' % str(time.time()-start_time))
    return m

def finish_op(config,op_config,dumper,from_scratch):
    """Update the config and backup model if needed

    :param config: the main configuration
    :param op_config: the optimizer config
    :param dumper: the dumping function (if needed)
    :param from_scratch: specifies if model was built from scratch
    :param pipeline: back up using pipeline protocol
    """
    model_out = 'model'
    name = ''

    if config.retrain_indv:
        name = 'model_indv_select'
        model_out = os.path.join(config.dir,'model_indv_select')
    elif config.retrain_temp:
        name = 'model_temp_select'
        model_out = os.path.join(config.dir,'model_temp_select')

    if not op_config.model_loc and not config.model_loc:
        model_out = os.path.join(config.dir,model_out)
        if dumper: dumper(model_out)

        op_config.model_loc = model_out
    elif config.retrain_indv or config.retrain_temp:
        if dumper: dumper(model_out)

    config.model_loc = op_config.model_loc
    config.restore = config.dir

    ## build run scripts
    if from_scratch:
        __eval_script(config.dir,"--eval_test",'test',config.model_loc)
        __eval_script(config.dir,"--eval_val",'valid',config.model_loc)
        #__eval_script(config.dir,"--eval_train",'train',config.model_loc)

    elif config.retrain_indv or config.retrain_temp:
        __eval_script(config.dir,"--eval_test",'test_%s' % name,model_out)
        __eval_script(config.dir,"--eval_val",'valid' % name,model_out)

# def pipeline_backup(optimizer,):
#     pass 
