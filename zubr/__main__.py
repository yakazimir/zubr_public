#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""
import sys
import traceback
import logging
from zubr.util.loader import load_module
from zubr import _heading as heading
from zubr.Pipeline import PipelineError

modes = {}
modes["alignment"]           = "zubr.Alignment"
modes["aligner"]             = "zubr.Alignment"
modes["align"]               = "zubr.Alignment"
modes["parser"]              = "zubr.Parse"
modes["pipeline"]            = "zubr.Pipeline"
modes["reranker"]            = "zubr.Reranker"
modes["paraphrase"]          = "zubr.Paraphrase"
modes["optimizer"]           = "zubr.Optimizer"
modes["feature_selector"]    = "zubr.FeatureSelection"
modes["alignment"]           = "zubr.Alignment"
modes["rankdecoder"]         = "zubr.RankDecoder"
modes["symmetricalignment"]  = "zubr.SymmetricAlignment"
modes["symalign"]            = "zubr.SymmetricAlignment"
modes["graphdecoder"]        = "zubr.GraphDecoder"
modes["doc_extractor"]       = "zubr.doc_extractor.DocExtractor"
modes["queryinterface"]      = "zubr.QueryInterface"
modes["queryserver"]         = "zubr.web.QueryServer"
modes["feature_extract"]     = "zubr.FeatureExtractor"

USAGE = """usage: python -m zubr mode [options]

current modes: %s""" % ', '.join(modes)

def main(argv):

    if (argv) and (argv[0].lower() in modes):
        mode = modes[argv[0].lower()]
    elif argv:
        exit('mode not known...\n %s' % USAGE)
    else:
        exit(USAGE)
    
    try:
        mod = load_module(mode)
        mod.main(argv)
        
    except PipelineError,e:
        print >>sys.stderr,"pipeline error encountered.."
        traceback.print_exc(file=sys.stdout)
        exit()
        
    except Exception, e:
        print >>sys.stderr,"uncaught error encountered.."
        traceback.print_exc(file=sys.stdout)
        exit() 

if __name__ == "__main__":
    main(sys.argv[1:])
