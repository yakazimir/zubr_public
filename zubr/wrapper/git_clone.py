# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Assumes that git is installed 

"""

import os
import sys
import subprocess

def git_clone(config):
    """The main function for running the git clone subprocess 

    :param config: the main configuration 
    """
    glog_path = os.path.join(config.dir,"git.log")
    out_dir = os.path.join(config.dir,config.git_name)
    args = "git clone %s %s" % (config.git_loc,out_dir)

    with open(glog_path,'w') as logger:
        p = subprocess.Popen(args,stdout=logger,stderr=logger,shell=True)
        p.wait()

    ## automatically add project path to configuration
      
    
def params():
    """The main parameters for running the git cloner
    
    :rtype: tuple
    """
    options = [
         ("--git_loc","git_loc","","str",
            "The location of the git project [default='']","GitWrapper"),
        ("--git_name","git_name","","str",
            "The name of the project [default='']","GitWrapper"),
    ]

    git_group = {"GitWrapper" : "Wrapper for using git clone"}
    return (git_group,options)


def main(config):
    """The main function for running the git clone wrapper

    :param config: the main configuration 
    """
    git_clone(config)

if __name__ == "__main__":
    main(sys.argv[1:])
