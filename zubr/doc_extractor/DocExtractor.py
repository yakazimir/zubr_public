# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""

import codecs
import os
import ast
import sys
import logging
from zubr.util import ConfigObj,ConfigAttrs
from zubr.doc_extractor.util import *

class DocExtractorBase(object):
    """Base class for extracting documentation"""

    @classmethod
    def load_project(cls,config):
        """Load an example project 

        :param config: the project or global configuration 
        """
        raise NotImplementedError

    def extract(self):
        """Extract the documentation 
        
        :rtype: None 
        """
        raise NotImplementedError

    def print_set(self):
        """Print the overall functions with documentation 

        :rtype: None 
        """
        raise NotImplementedError 

    @property
    def logger(self):
        ## instance logger
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level)

class PyExtractor(DocExtractorBase):
    
    """Uses ast to extract documentation from functions, classes, and modules"""

    def __init__(self,cfiles,config):
        """Initialize a py extractor instance 


        :param cfiles: paths to the files to be processed 
        :param config: the extractor local configuration
        """
        self.files  = cfiles
        self.config = config

    def extract(self):
        """Main method for extracting the target documentation.

        :rtype: None
        """
        ## do the extraction, print if specified 
        py_extract(self.files,self.config)

    @classmethod
    def load_project(cls,config):
        """Load an example project 

        :param config: the project or global configuration 
        """
        settings = ConfigAttrs()
        files = load_python(config,settings)
        return cls(files,settings)
    
class JavaExtractor(DocExtractorBase):
    pass

EXTRACTORS = {
    "py" : PyExtractor,
}

    
def DocExtractor(config):
    """Factory for building a document extractor

    :param config: the main configuration 
    """
    extractor = EXTRACTORS.get(config.doc_extractor,None)
    if not extractor:
        raise ValueError('Uknown extractor type: %s' % extractor)
    return extractor

### CLI STUFF

def params():
    """Main parameters for running the doc extractor

    """
    groups = {}
    groups["DocExtractor"] = "General settings for the document extractor"
    groups["PyExtractor"] = "General settings for python extractor (if used)"

    options = [
        ("--proj","proj","","str",
         "The location of the target project [default='']","DocExtractor"),
        ("--proj_name","proj_name","","str",
         "The name of the project [default='']","DocExtractor"),
        ("--src_loc","src_loc","src/","str",
         "Directory in project where source code is [default='src/']","DocExtractor"),
        ("--doc_extractor","doc_extractor","py","str",
         "The type of extractor to use [default='python']","DocExtractor"),
        ("--ignore_magic","ignore_magic",False,"bool",
         "Ignore magic methods [default=False]","PyExtractor"),
        ("--ignore_test","ignore_test",True,"bool",
         "Ignore methods/classes related to testing [default=True]","DocExtractor"),
        ("--extract_undoc","extract_undoc",False,"bool",
         "Ignore undocumented functions [default=False]","DocExtractor"),
        ("--dir_blocks","dir_blocks","","str",
         "src directories to block (delimited by +) [default='']","DocExtractor"),
        ("--max_args","max_args",4,"int",
         "The maximum number of arguments [default=5]","DocExtractor"),
        ("--preproc","preproc",True,"bool",
         "Run simple preprocessor on text [default=True]","DocExtractor"),
        ("--print_data","print_data",True,"bool",
         "Print the data in the end [default=True]","DocExtractor"),
        ("--out_dir","out_dir",'',"str",
         "The place to print data to [default='']","DocExtractor"),
        ("--run_exp","run_exp",True,"bool",
         "Run an experiment on extracted data [default=True]","DocExtractor"),
        ("--prepare_fun","prepare_fun",True,"bool",
         "Preprocess the function reprensetation [default=True]","DocExtractor"),
        ("--web_addr","web_addr",'',"str",
         "The web address of project (if one exists) [default='']","DocExtractor"),
        ("--class_info","class_info",False,"bool",
         "Extract class information to use as features [default=False]","DocExtractor"),
        ("--online_addr","online_addr",'',"str",
         "Online source code address (if available) [default='']","DocExtractor"),
    ]
    return (groups,options)

def argparser():
    """Returns an argument parser to be used for this module

    :returns: default argument parser
    """
    usage = """python -m zubr doc_extractor [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage)
    return argparser


def main(argv):
    """The main execution point 


    :param argv: the cli input or main configuration
    :rtype: None
    """
    if isinstance(argv,ConfigAttrs):
        config = argv
    else:
        parser = argparser()
        config = parser.parse_args(argv[1:])
        logging.basicConfig(level=logging.DEBUG)

    eclass = DocExtractor(config)

    ## build dataset
    extractor = eclass.load_project(config)
    extractor.extract()

if __name__ == "__main__":
    main(sys.argv[1:])
