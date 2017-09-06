#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Utility functions for the pipeline code

"""
import os
import sys
import subprocess


PATH = os.path.abspath(os.path.dirname(__file__))
ZUBR_PATH = os.path.abspath(os.path.join(PATH,"../../"))

def make_run_script(config,cli_input):
    """Make a script for running pipeline given original cli input

    :param config: the main configuration 
    :param launch_loc: the location of the pipeline code 
    :param cli_input: the command line input 
    :rtype: None
    """
    full = "cd %s\n./run_zubr %s" % (ZUBR_PATH,cli_input)
    script_path = os.path.join(config.dir,"run.sh")
    restore = os.path.join(config.dir,"restore.sh")
    if not os.path.isfile(restore):
        with open(script_path,'w') as my_script:
            print >>my_script,full

        subprocess.call(['chmod', '755', script_path])

def make_restore(config,argv,level):
    """Make a script for rerunning at the point last stopped

    """
    full = "cd %s\n./run_zubr %s --dir %s --start %d" % (ZUBR_PATH,' '.join(argv[:2]),config.dir,level)
    script_path = os.path.join(config.dir,"restore.sh")
    with open(script_path,'w') as my_script:
        print >>my_script,full
    subprocess.call(['chmod','755',script_path])
