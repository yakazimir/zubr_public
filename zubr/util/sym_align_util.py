# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Utilities for working with alignmetn models 

"""
import os
import sys
import re

def ftoe_config(main_config):
    """Setup the configuration for the ftoe aligner 

    :param main_config: the overall experiment configuration 
    :returns: copy of main configuration with modified values
    :rtype: zubr.util.config.ConfigAttrs
    """
    ftoe_config = main_config.copy()
    ftoe_config.aligntraining = False
    return ftoe_config

def etof_config(main_config):
    """Setup the configuration for the etof aligner 

    :param main_config: the overall experiment configuration 
    """
    orig_source = main_config.source
    orig_target = main_config.target
    etof_config = main_config.copy()
    
    ## reverse the suffix names 
    etof_config.source = orig_target
    etof_config.target = orig_source
    etof_config.aligntraining = False
    etof_config.train2 = True

    return etof_config

def print_phrase_table(frev,erev,phrase_table,config):
    """Printed phrases table entries 

    :param frev: reverse dictionary for looking up foreign word ids
    :type frev: dict
    :param erev: reverse dictionary for looking up english word ids
    :type erev: dict
    :param phrase_table: the list of phrase pairs
    :type phrase_table: dict
    :rtype: None    
    """

    ## decode on output 
    if config.dir:
        out = open(os.path.join(config.dir,"phrase_table.txt"),'w')
    else:
        out = sys.stdout

    ## iterate through phrases 
    for ((foreign,english),freq) in phrase_table.items():
        foreign_str = ' '.join([frev[i] for i in foreign])
        english_str = ' '.join([erev[i] for i in english])
        print >>out, "%s\t%s\t%f" % (english_str.encode('utf-8'),
                                         foreign_str.encode('utf-8'),freq)
    if config.dir:out.close()
