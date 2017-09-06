#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

from __future__ import print_function, absolute_import
from zubr._version import __version__

__author__ = "Kyle Richardson"
__author_email__ = "kazimir.richardson@gmail.com" 
__maintainer__ = __author__
__maintainer_email__ = __author_email__
__license__ = "GPL2"
__url__ = "www.github.com/yakazimir/zubr"
__classifiers__ = []

__keywords__ = [
    "NLP",
    "Computational Linguistics",
    "Parsing",
    "Computational Semantics",
    "PCFGs",
    "Machine Translation",
]

__copyright__ = """\
Copyright (C) 2017 Kyle Richardson

see LICENSE for additional terms and conditions
"""

_heading = """Zubr: A Semantic Parsing toolkit, version %s,
Copyright (C) 2016 Kyle Richardson.

Zubr comes with ABSOLUTELY NO WARRANTY; This is free software,
and you are welcome to redistribute it under certain conditions.
See LICENSE distributed with the source code for more
information""" % __version__

from zubr.util.loader import load_module

def get_config(zubr_module):
    """Return back a configuration instance for a utility with default values 

    >>> from zubr import get_config 
    >>> aligner_config = get_config("zubr.Alignment") 
    >>> aligner_config.aiters
    5

    :param zubr_module: pointer to the target configuration 
    :returns: a default configuration object for input module 
    :rtype: zubr.util.config.ConfigAttrs
    """
    mod = load_module(zubr_module)
    if hasattr(mod,"argparser"):
        config_parser = mod.argparser()
        return config_parser.get_default_values()
    raise ValueError('No argparser for module: %s' % zubr_module)

from zubr.util.config import ConfigAttrs
empty_config = ConfigAttrs()

import os
src_loc = os.path.os.path.abspath(os.path.dirname(__file__))
lib_loc = os.path.abspath(os.path.join(src_loc,"../"))
