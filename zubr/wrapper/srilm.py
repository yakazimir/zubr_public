# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson


a wrapper for building srilm language model files

"""

import re
import time
import os
import logging
import subprocess
from zubr import src_loc
from zubr.util.config import ConfigAttrs


OPS = {
    "" : None,
}

class SRILMSubprocess(object):

    """SRILM subprocess, for running and training language model"""

    def __init__(self,srilm_loc,srilm_platform):
        """ 

        :param srilm_loc: the location of the srilm library 
        :param srilm_platform: the platform being run on 
        """
        self.loc       = srilm_loc
        self.platform  = srilm_platform

    def build_model(self,config):
        """Build a language model from text input 
        
        :param data_path: the path to the training data 
        """
        ## setting specific settings 
        srilm_bin = os.path.join(config.srilm_loc,"bin")
        platform_bin = os.path.join(srilm_bin,config.srilm_platform)
        ncounter = os.path.join(platform_bin,"ngram-count")
        ##
        
        ## log path 
        srilm_log_path = os.path.join(config.dir,"srilm.log")

        ##lmout
        out_path = os.path.join(config.dir,config.lm_out)

        ## options
        options = " " if not config.srilm_set else config.srilm_set
        
        ## just add to log if file already exists
        if os.path.isfile(srilm_log_path): ptype = 'a'
        else: ptype = 'w'

        ## run the language model 
        with open(srilm_log_path,ptype) as logger:
            args = "./%s %s -lm %s -text %s" % (ncounter,options,out_path,config.text)
            p = subprocess.Popen(args,stdout=logger,stderr=logger,shell=True)
            p.wait()

    def __gzip_lm_file():
        """Backup the language model file to gzip

        """
        pass

def params():
    """Main parameters for running the srilm stuff

    :returns: touple of parameter descriptions and parameter values
    """
    default_srilm = os.path.join(src_loc,"srilm")

    options = [
        ("--srilm_loc","srilm_loc","zubr/srilm","str",
            "Location of the srilm file [default='']","SRILMWrapper"),
        ("--srilm_platform","srilm_platform","i686-m64","str",
             "Location of the srilm file [default='']","SRILMWrapper"),
        ("--srilm_set","srilm_set","","str",
             "Settings for the srilm as a string [default='']","SRILMWrapper"),
        ("--lm_out","lm_out","lm.srilm","str",
             "Name of the output language model [default='lm.srilm']","SRILMWrapper"),
        ("--text","text","","str",
             "The target data to train language model with [default='']","SRILMWrapper"),
    ]

    group = {"SRILMWrapper" : "Settings for the srilm toolkit"}

    return (group,options)

def argparser():
    """Return an aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr srilm [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser


def main(argv):
    """Main execution point for running the srilm toolkit 

    :param config: the main configuration 
    """
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])
        logging.basicConfig(level=logging.DEBUG)

    if not config.text:
        exit('Please specify language model training data!!')

    lm = SRILMSubprocess(config.srilm_loc,config.srilm_platform)
    lm.build_model(config)


def __main__(self):
    main(sys.argv[1:])
