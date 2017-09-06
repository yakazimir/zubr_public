# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package. 

author : Kyle Richardson

A set of useful decorators used in zubr 
"""

import sys
import functools
import traceback

__all__ = ["catchPipelineError"]

def catchPipelineError(exception=Exception):
    """decorator for debugging errors in pipeline script"""
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                result = f(*args, **kwargs)
            except Exception,e:
                print >>sys.stderr,"error in pipeline script..."
                traceback.print_exc()
                sys.exit()
            else: 
                return result
        return wrapper
    return deco

if __name__ == "__main__":
    my_function(1)
