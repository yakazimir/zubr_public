#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

setup for zubr lisp 
"""

from zubr.util.config import ConfigAttrs,ConfigObj

def params():
    """main parameters for running zubr lisp

    :returns: tuple of module description plus command line settings
    :type: tuple
    """
    options = [
        ("--repl","repl",False,"bool",
         "run repl/zubr shell on invocation [default=True]","ZubrLisp"),
        ("--script","script",'',"str",
         "input file to parse [default=True]","ZubrLisp"),
        ("--test","test",'',"str",
         "run a lisp test [default='']","ZubrLisp"),
        ("--benchmark","benchmark",False,"bool",
         "benchmark lisp core functions to python [default=False]","ZubrLisp"),                           
         ]

    group = {"ZubrLisp":"Zubr lisp settings and defaults"}
    return (group,options)

def argparser():
    """returns a zubr lisp argument parser

    :type: zubr.util.config.ConfigObj
    """
    usage = """python -m zubr.zubr_lisp [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage)
    return argparser

def main(argv):
    """main execution function for setting up zubr lisp configuration

    :param argv: command line input
    :type argv: str
    """
    config_parser = argparser()
    config = config_parser.parse_args(argv[1:])
    return config
    
    

