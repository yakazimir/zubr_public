# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

This script is used for running a rank decoder with some data
 
Arguments to run this script: 

     --dir : Directory to forward all output, store files,...

     --atraining : This points to the data directory and name of the target dataset, 
     e.g., $ZUBR/experiments/technical_documentation/data/elisp/elisp, where ``elisp`` 
     is the name of the dataset. Including in this directory should be the following: 

          {name}.{e,f}       : The main training data 
          {name}_bow.{e,f}   : Main training without extra data (optional, use --bow switch to turn on) 
          {name}_valid.{e,f} : The validation set (optional) 
          {name}_test.{e,f}  : The testing set (optional) 
          rank_list.txt      : The list of all target representations (required) 

     --decoder : The type of decoder to use (e.g., IBM Model, neural network model,...)

     --help : To see all options, type ./run_zubr pipeline bin/rank_decoder --help

What the rank decoder does is trains a translation model from parallel data, then 
uses this model to rank items in the rank list given each input. As such, it can be
fairly slow for large datasets and rank_lists, but it useful for trying out new 
translation models to see if they work.  

"""

import os
import shutil

params = [
     ("--data","data","","str",
     "Location of data directory [default='']","RankDecoder"),
     ("--bow","bow",False,"bool",
     "Use the bag of word data [default=False]","RankDecoder"),
]

description = {"RankDecoder" : "settings for the rank decoder"}

tasks = [
    "setup_data",
    "zubr.RankDecoder"
]


def __check_data(data_pos):
    """check that the given data exists

    :param data_pos: the target data directory 
    """
    if not data_pos:
        raise ValueError('Must specify working directory and dataset name!')
    wdir = os.path.dirname(data_pos)
    if not os.path.isdir(wdir):
        raise ValueError('Unknown target directory!')
    
def setup_data(config):
    """Set up the data needed to run a rank decoder, instantiate config values

    :param config: the main configuration 
    :raises: ValueError 
    """
    __check_data(config.atraining)

    ## copy over the rank file and set config 
    rfile = os.path.join(os.path.dirname(config.atraining),"rank_list.txt")
    shutil.copy(rfile,config.dir)
    config.rfile = rfile

    ## copy over the main data
    name_prefix = config.atraining if not config.bow else config.atraining+"_bow"
    data_name = os.path.basename(config.atraining)
    ename = os.path.join(config.dir,data_name+".%s" % config.target)
    fname = os.path.join(config.dir,data_name+".%s" % config.source)
    english_side = name_prefix+".%s" % config.target
    foreign_side = name_prefix+".%s" % config.source
    shutil.copy(english_side,ename)
    shutil.copy(foreign_side,fname)
    

    ## copy the development data (if it exists)
    ename = os.path.join(config.dir,data_name+"_val.%s" % config.target)
    fname = os.path.join(config.dir,data_name+"_val.%s" % config.source)
    evalid = config.atraining+"_val.%s" % config.target
    fvalid = config.atraining+"_val.%s" % config.source 

    ## copy it over
    shutil.copy(evalid,ename)
    shutil.copy(fvalid,fname)

    config.atraining = os.path.join(config.dir,data_name)
            

    ## copy the testing data (if it exists)
        
