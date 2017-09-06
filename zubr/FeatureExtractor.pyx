# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Select feature extractors 

"""

import os
import time
import traceback
import sys
import logging
import time
from zubr.AlignExtract cimport AlignerExtractor
from zubr.BowExtract cimport BowExtractor
from zubr.GraphExtractor cimport GraphRankExtractor
from zubr.util import ConfigObj,ConfigAttrs
from zubr.Dataset cimport RankDataset,RankPair
from zubr.Extractor cimport Extractor

## list of available feature extractors
## append list to add new extractor

EXTRACTORS = {
    "aligner": AlignerExtractor,
    "bow"    : BowExtractor,
    "graph"  : GraphRankExtractor,
}

def ExtractorClass(etype):
    """Factory method for returning a extractor type

    :param etype: the type of extractor
    :type etype: basestring
    :returns: the desired extractor class 
    """
    if etype not in EXTRACTORS:
        raise ValueError('Unknown extractor type: %s' % etype)
    return EXTRACTORS[etype]

## c methods

cdef int extract_from_dataset(Extractor extractor,
                                  RankDataset dataset,str dtype) except -1:
    """Extract features from file

    :param dataset: the dataset to extract from 
    :param extractor: the feature extractor 
    :param dtype: the data type
    """
    cdef int size = dataset.size,data_point
    cdef RankPair instance

    stime = time.time()
    extractor.logger.info('Started the extraction loop...')

    for data_point in range(size):
        instance = dataset.get_item(data_point)
        extractor.extract(instance,dtype)

    extractor.logger.info('Finished extraction in %s seconds' % str(time.time()-stime))
        
### CLI INFORMATION

def params():
    """defines the main parameters for the extractors

    :rtype: tuple
    :returns: description of switches for these utilities 
    """
    from zubr.AlignExtract import params
    from zubr.BowExtract import params as bparams
    from zubr.GraphExtractor import params as gparams
    
    group,ae_params = params()
    bow_group,bow_params = bparams()
    ggroup,g_params = gparams()
    group["Extractor"] = "settings for setting up extractor"
    group.update(bow_group)
    group.update(ggroup)
    
    options = [
        ("--extractor","extractor",'aligner',"str",
        "The type of extractor to use [default='aligner']","Extractor"),
        ("--extractor_job","extractor_job",'',"str",
        "A particular extraction job to run [default=False]","Extractor"),
        ("--pipeline_backup","pipeline_backup",False,"bool",
        "Backup using the pipeline backup [default=False]","Extractor"),
        ("--elog","elog",'',"str",
        "Will log to a file rather than to stdout [default='']","Extractor"),
        ("--epath","epath",'',"str",
        "The path or working directory for extractor [default='']","Extractor"),
        ("--doffset","doffset",0,int,
        "The dataset offset, if running extractor on data slice [default=0]","Extractor"),
    ]

    options += ae_params
    options += bow_params
    options += g_params
    return (group,options)

def argparser():
    """Return an aligner argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    
    usage = """python -m zubr feature_extract [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

def main(config):
    """main execution function for setting up a feature extractor

    :param config: zubr configuration object 
    :type config: zubr.util.config.ConfigAttrs
    """
    if isinstance(config,ConfigAttrs):
        config = config
    else:
        parser = argparser()
        config = parser.parse_args(config[1:])
        if config.elog: 
            logging.basicConfig(filename=config.elog,level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)

    load_util = logging.getLogger('zubr.FeatureExtractor.main')
    
    try:
        
        ## build an ordinary extractor 
        if not config.extractor_job:
            extractor_class = ExtractorClass(config.extractor)
            extractor = extractor_class.build_extractor(config)
            
        ## run an extractor on some target data
        else:
            ## load a backup of the extractor
            load_util.info('Building the extractor')

            ## rebuild the configuration
            wdir = config.epath
            config.restore_old(wdir)
            config.dir = wdir
            
            ## extractor class and instance 
            extractor_class = ExtractorClass(config.extractor)
            extractor = extractor_class.load_backup(config)
            extractor.dir    = wdir
            extractor.offset = config.doffset
            extractor.logger.info('Loading extracted...')

            # # ## build the dataset
            dpath = os.path.join(config.dir,"%s.data" % config.extractor_job)
            dataset = RankDataset.load(dpath)
            dataset.logger.info('Loaded the dataset...')
            
            ## finally, do the extraction
            extract_from_dataset(extractor,dataset,config.extractor_job)

    except Exception,e:
        load_util.error(e,exc_info=True)
        traceback.print_exc(file=sys.stdout)

    finally:
        if config.pipeline_backup:
            extractor.backup(config.dir)
        else: 
            if config.dump_models:
                model_out = os.path.join(config.dir,"extractor.p")
                extractor.dump(model_out)
                
            base_model = os.path.join(config.dir,"base.model")
            if config.cleanup and os.path.isfile(base_model):
                os.remove(base_model)

        ## exit the extractor
        extractor.exit()
        
