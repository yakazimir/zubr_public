#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

import re
import os
import sys
import time
import codecs
import logging
from zubr.Datastructures import ParallelCorpusSuffixArray

### REGEXES for preprocessing

short    = re.compile(r'(e\.g\.?|i\.\e\.?|ect\.+)')
paren1   = re.compile(r'\(([a-zA-Z0-9\s\-\+\.\;\:\,]+)\)')
paren2   = re.compile(r'\[([a-zA-Z0-9\s\-\+\.]+)\]')
punc1    = re.compile(r'\s(\,|\)|\(|\?)\s')
punc2    = re.compile(r'\s(\,|\)|\(|\?|\.)$')
punc3    = re.compile(r'(\?|\!|\.|\;|\n|\\n)$')
quote1   = re.compile(r'\'([a-zA-Z\s\-]+)\'')
quote3   = re.compile(r'\"([a-zA-Z\s\-\!]+)\"')
quote2   = re.compile(r'\`|\'|\"+')
comma    = re.compile(r'([a-zA-Z0-9])(\,|\;|\:)')
minus    = re.compile(r' \- ')

## utility for building paraphrase model

plogger = logging.getLogger('zubr.util.paraphrase_util')

def build_suffix(config):
    if not config.atraining:
        raise ValueError('no parallel data given!')

    datasets = [] 

    if isinstance(config.atraining,basestring):
        config.atraining = config.atraining.split('+')

    ## go through each dataset 3
    for dset in config.atraining:
        dname = dset.split('/')[-1]

        edata = os.path.join(config.dir,'%s/target.txt' % dname)
        fdata = os.path.join(config.dir,'%s/source.txt' % dname)
        align_data = os.path.join(config.dir,'%s_alignment.txt' % dname)

        ## open edata 
        with codecs.open(edata,encoding='utf-8') as edata:
            eside = [l.strip().split() for l in edata.readlines()]

        ## open fdata
        with codecs.open(fdata,encoding='utf-8') as fdata:
            fside = [l.strip().split() for l in fdata.readlines()]

        ## open alignment data
        with codecs.open(align_data,encoding='utf-8') as adata:
            aligned = [i.strip() for i in adata.readlines()]
                        

        assert len(eside) == len(fside),'datasets do not match!'
        assert len(eside) == len(aligned),'datasets do not match!'

        ## add to collection
        plogger.info('building suffix array datastructure for %s dataset' % dname)
        t = time.time()
        sarray = ParallelCorpusSuffixArray.build_array(dname,eside,fside,aligned,
                                                       dir=config.dir,
                                                       N=config.parallel)

        plogger.info('finished building suffix array in %s seconds' % str(time.time()-t))
        datasets.append(sarray)

    return datasets


def clean_examples(sentence):
    """some light preprocessing for the paraphrase data

    :param sentence: the sentence to clean
    """
    cleaned = sentence.strip().lower()
    cleaned = cleaned.split('. ')[0]

    cleaned = re.sub(paren1,r'\1',cleaned)
    cleaned = re.sub(paren2,r'\1',cleaned)
    cleaned = re.sub(r'\.$','',cleaned)
    cleaned = re.sub(punc1,'',cleaned)
    cleaned = re.sub(punc2,'',cleaned)
    cleaned = re.sub(punc3,'',cleaned)
    cleaned = re.sub(quote1,'',cleaned)
    cleaned = re.sub(quote2,'',cleaned)
    cleaned = re.sub(comma,r'\1',cleaned)
    cleaned = re.sub(minus,' ',cleaned)
    cleaned = re.sub(r'^\- ',' ',cleaned)
    cleaned = re.sub(r'^\:\s*$','',cleaned)
    cleaned = re.sub(r'^(_|\(|\)|\,|\.)\s*$','',cleaned)
    cleaned = re.sub(r'\- ',' ',cleaned)
    cleaned = re.sub(r'^\<\>\s*$','',cleaned)
    cleaned = re.sub(r'\.\s*$','',cleaned)
    cleaned = re.sub(r'\; ',' ',cleaned)
    cleaned = re.sub(r'\s+',' ',cleaned)

    return cleaned.strip()
            

def preprocess_data(config):
    """Preprocess the data

    :param config: the main configuration
    """
    data_directory = os.path.join(config.dir,"data")
    os.mkdir(data_directory)
    newords = []
    nfwords = []

    for num,dataset in enumerate(config.atraining):
        english = "%s.e" % dataset
        foreign = "%s.f" % dataset
        num_ewords = 0
        num_fwords = 0

        base_name = dataset.split('/')[-1]
        new_e = os.path.join(data_directory,"%s.e" % base_name)
        new_f = os.path.join(data_directory,"%s.f" % base_name)
        base_path = os.path.join(data_directory,base_name)
        
        es = codecs.open(english,encoding='utf-8').readlines()
        fs = codecs.open(foreign,encoding='utf-8').readlines()
        
        dlen = len(fs)

        ## check that they are the same length 
        assert len(es) == len(fs) == dlen, 'mismatching parallel data!'

        with codecs.open(new_e,'w',encoding='utf-8') as english_new:
            with codecs.open(new_f,'w',encoding='utf-8') as foreign_new: 
            
                for k in range(dlen):
                    esen = es[k]
                    fsen = fs[k]
                    esen = clean_examples(esen)
                    fsen = clean_examples(fsen)
        
                    if not esen or not fsen or len(esen) == 1 or len(fsen) == 1:
                        continue

                    if len(esen.split()) > 80 or len(fsen.split()) > 80:
                        continue 
                    
                    # print >>english_new,esen.strip()
                    # print >>foreign_new,fsen.strip()
                    print >>english_new,esen.strip().lower()
                    print >>foreign_new,fsen.strip().lower()
                    
                    num_ewords += len(esen.split())
                    num_fwords += len(fsen.split())

        config.atraining[num] = base_path
        newords.append(num_ewords)
        nfwords.append(num_fwords)

    with codecs.open(os.path.join(config.dir,'INFO_COUNTS'),'w') as my_info:
        print >>my_info,"# English words seen: %s" % sum(newords)
        print >>my_info,"# Foreign words seen: %s" % sum(nfwords)


