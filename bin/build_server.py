import os
from shutil import copytree,copy

## parameters
params = [
]

description = {}

tasks = [
    "zubr.doc_extractor.DocExtractor",
    "setup_aligner_settings",
    "zubr.SymmetricAlignment",
    "swap_data",
    "zubr.Dataset",
    "zubr.FeatureExtractor",
    "zubr.Optimizer",
    "before_interface",
    "zubr.QueryInterface",
]

def swap_data(config):
    """change to ranking data to files with name_rank.{e,f}"""
    reranke = "%s_bow.e" % config.atraining
    rerankf = "%s_bow.f" % config.atraining
    etrain = "%s.e" % config.atraining
    ftrain = "%s.f" % config.atraining
    copy(reranke,etrain)
    copy(rerankf,ftrain)

def setup_aligner_settings(config):
    config.atraining = os.path.join(config.dir,"data")
    config.rfile = os.path.join(config.dir,"rank_list.txt")

    ## extract phrase table 
    config.extract_phrases = True
    config.print_table     = True

    ## if you want to use hiero phrase features
    ## put glue and hiero grammar rules in config path
    config.extract_hiero   = True
    if config.extract_hiero:
        config.hierogrammar = os.path.join(config.dir,"hiero_rules.txt")
        config.gluegrammar = os.path.join(config.dir,"grammar.txt")

    config.rmodel = 'symaligner'
    config.dump_models = True
    config.cleanp = True

def before_interface(config):
    ## point to the built model for building a query interface 
    config.qmodel = os.path.join(config.dir,"model")
