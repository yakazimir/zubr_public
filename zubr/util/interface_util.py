# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson


"""

import os
import logging 
import codecs
import numpy as np

__all__ = [
    "read_rank_file",
]

alogger = logging.getLogger('zubr.util.interface_util')

def read_rank_file(config,rtype):
    """Parses the representation file for the rank query interface 

    :param config: the experiment or interface configuration
    :param rtype: the type of rank interface
    :raises: ValueError 
    :returns: a list of items from the rank list provided 
    :rtype: np.ndarray 
    """
    rfile = 'rank_list.txt' if rtype == 'str' else 'rank_list_uri.txt'
    full_path = os.path.join(config.dir,rfile)
    
    ## check that file exists 
    if not os.path.isfile(full_path):
        raise ValueError('Cannot find rank file!: %s' % full_path)

    ritems = []
    
    with codecs.open(full_path,encoding='utf-8') as my_ranks:
        for line in my_ranks:
            line = line.strip()
            ## tab delimited?
            tspaced = line.split('\t')
            if len(tspaced) == 1:
                if rtype == 'html':
                    raise ValueError('Rank list for html badly formatted!')
                ritems.append(line.strip())
            else:
                ritems.append(tspaced)
    return np.array(ritems,dtype=np.object)
