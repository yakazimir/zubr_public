import os
from zubr.util.decor import catchPipelineError
from shutil import copytree,copy

params = [
        ("--amax","amax",90,int,
         "maximum sentence length [default=90]","BOWScript"),
]
    
description = {"BOWScript": "settings for bag of word script"}

tasks = [
    "setup_pipeline",           ## set up config
    "swap_data",                ## swap in training data for reranker 
    "zubr.Dataset",             ## build dataset for reranker
    "zubr.FeatureExtractor",    ## build feature extractor
    "zubr.Optimizer",           ## reranker optimizer
]

@catchPipelineError()
def setup_pipeline(config):
    """set up the base model for the reranker

    :param config: main config object for experiment and related utilities
    :type config: zubr.util.config.ConfigAttrs
    :raises ValueError: raises when model is selected that is not supported
    :rtype: None 
    """

    ## check config values 
    config.dump_models = True
    config.cleanup     = True
    config.rmodel      = 'bow'
    config.extractor   = 'bow'
    config.atraining   = config.bow_data
    config.source = 'f'
    config.target = 'e'

    ## swap the bow data as the real data
    data_dir = os.path.dirname(config.atraining)
    new = os.path.join(config.dir,"orig_data")
    name = config.atraining.split('/')[-1]
    if config.dir != data_dir or not os.isdir(new):
        copytree(data_dir,new)
        config.atraining = os.path.join(new,name)
        config.rfile = os.path.join(config.dir,"orig_data/rank_list.txt")

    basee = "%s_bow.e" % config.atraining
    basef = "%s_bow.f" % config.atraining
    etrain = "%s.e" % config.atraining
    ftrain = "%s.f" % config.atraining
    copy(basee,etrain)
    copy(basef,ftrain)
        
@catchPipelineError()
def swap_data(config):
    """Make sure <data>_bow is the main data"""
    pass 
    # basee = "%s_bow.e" % config.atraining
    # basef = "%s_bow.f" % config.atraining
    # etrain = "%s.e" % config.atraining
    # ftrain = "%s.f" % config.atraining
    # copy(basee,etrain)
    # copy(basef,ftrain)

