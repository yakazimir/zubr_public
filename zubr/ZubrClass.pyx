# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""
import os
import logging 
import gzip
import bz2
import time 
from zubr import lz4tools

## try to use cPickl
import pickle

cdef class ZubrClass:
    pass 

cdef class ZubrLoggable(ZubrClass):

    @property
    def logger(self):
        """Returns a logger instance"""
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level)

cdef class ZubrSerializable(ZubrLoggable):

    def dump(self,path):
        """Pickle the given instance 

        :param path: path to put item 
        :type path: str
        :rtype: None 
        """        
        self.logger.info('pickling the object')
        if not '.gz' in path: path +=".gz"
        st = time.time()
        with gzip.open(path,'wb') as my_path:
            pickle.dump(self,my_path)
        self.logger.info('pickled in %s seconds' % str(time.time()-st))

    def dump_large(self,path):
        """Uses lz4 to compress file due to problem with gzip

        :param path: the file base path name 
        :type path: str
        :rtype: None
        """
        base_path = path
        st = time.time()

        ## pickle the file (as normal)
        with open(path,'wb') as my_path: pickle.dump(self,my_path)
        ## compress file with lz4
        lz4tools.compressFileDefault(base_path)
        ## remove large file
        os.remove(base_path)
        self.logger.info('backed up and compressed in %s seconds' % str(time.time()-st))

    @classmethod
    def load(cls,path):
        """Load a pickled instance of object 

        :param path: path to pickled instance 
        :type path: str 
        :rtype: None 
        """
        if '.gz' not in path: path += ".gz"
        with gzip.open(path,'rb') as my_instance:
            return pickle.load(my_instance)

    @classmethod
    def load_large(cls,path):
        """Loads a large pickled object with lz4 

        :param path: the file base path name 
        :type path: str
        :returns: class instance
        """
        lz4path = path
        if '.lz4' not in path: lz4path += ".lz4"

        ## decompress file with lz4
        lz4tools.decompressFileDefault(lz4path)
        ## open uncompressed file
        with open(path,'rb') as my_instance:
            instance = pickle.load(my_instance)
            
        ## delete uncompressed file
        os.remove(path)
        return instance 

    def __reduce__(self):
        ## pickle implementation
        raise NotImplementedError

cdef class ZubrConfigurable(ZubrSerializable):

    @classmethod
    def from_config(self,config):
        """Creates a class instance from a zubr configuration

        :param config: the configuration
        """
        raise NotImplementedError
