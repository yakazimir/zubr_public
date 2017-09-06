#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""
from datetime import datetime
from shutil import rmtree
import os
from zubr.util._util_exceptions import ZubrUtilError
#from zubr import gen_config

ZUBR_TOP = os.path.abspath(os.path.dirname(__file__)+"../../../")
ZUBR_EXP = os.path.join(ZUBR_TOP,"examples/results") 


## functions for setting up experiments 

__all__ = ["dump_config","print_readme","setup_from_config",
           "make_experiment_directory"]

class OsUtilError(ZubrUtilError):
    pass 

def print_readme(readme_msg,path):
    """print readme message to file 

    :param readme_msg: readme message
    :type readme_msg: str
    :param path: path to print readme file
    :type path: str
    :rtype: None
    """
    readme_path = os.path.join(path,"README.txt")
    with open(readme_path,'w') as readmeout:
        print >>readmeout,readme_msg

def write_shell_script(config):
    pass

def make_experiment_directory(path='',config=None):
    """build a new directory from path, if
    no place is provided, create a random directory inside
    zubr source examples/results directory

    :param path: desired location of new directory (possibly none) 
    :type path: str
    :returns: full path of new directory
    :rtype: str
    :raises: OsUtilError
    """
    directory = path
    
    if not path:
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        directory = os.path.join(ZUBR_EXP,timestamp)

    directory = os.path.abspath(directory) 
    if os.path.isdir(directory) and not config.override:
        raise OsUtilError(
            'directory already exists, use --override option: %s'
            % directory)
    elif os.path.isdir(directory): 
        rmtree(directory)
        
    os.makedirs(directory)
    return directory
