#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

import sys
import os
import copy
import ConfigParser
import yaml
import logging
from optparse import OptionParser,OptParseError,OptionGroup,Values 
from yaml.error import YAMLError

__all__ = ["ConfigObj","ErrorReadingConfig",
           "ErrorParsingConfig","ConfigAttrs"]

util_logger = logging.getLogger('zubr.util.config')


class ErrorReadingConfig(OptParseError):
    """Config file error """
    pass

class ErrorParsingConfig(ErrorReadingConfig):
    """Error parsing the contents of yaml file"""
    pass 

class ConfigAttrs(Values):
    """configuration attribution object, subclass of optparse
    Values object. 
    """

    def __init__(self,defaults=None):
        if isinstance(defaults,dict):
            for (attr,val) in defaults.items():
                setattr(self,attr,val) 

    def print_to_yaml(self,path):
        """print config object state to yaml file"""
        with open("%s/config.yaml" % path,'w') as my_yaml:
            print >>my_yaml,self.__str__()

    def add_attr(self,attr,value):
        setattr(self,attr,value)

    def restore_old(self,path,ignore=[]):
        """updates using an older yaml config file"""
        
        try:
            with open("%s/config.yaml" % path) as old:
                config_values =  yaml.safe_load(old)
                for (key,val) in config_values.items():
                    if key not in ignore: 
                        self.add_attr(key,val)
        except IOError,e:
            util_logger.error(e,exc_info=True)
            raise ErrorReadingConfig('error with old config: %s' % path)


    def copy(self):
        """Creates a deep copy of this configuration object 

        :returns: deep copy
        """
        new = ConfigAttrs()
        for key,val in vars(self).items():
            new.add_attr(key,val)
        return new
                
    ## custom pickle implementation
        
    def __getinitargs__(self):
        return [self.__dict__]

    def __getstate__(self):
        return self.__dict__

    def __reduce__(self):
        return ConfigAttrs,()

    def __str__(self):
        r_str = ''
        for key,val in vars(self).items():
             r_str += "%s : %s\n" % (key,val)
        return r_str

    def __getattr__(self,x):
        try:
            Values.__getattr__(x)
        except AttributeError:
            return None

    def __eq__(self,other):
        if not isinstance(other,type(self)):
            return False
        return vars(self) == vars(other)

    def __nonzero__(self):
        return True 
    
    def __neq__(self,other):
        return not (self == other) 
    
    def __repr__(self):
        return "<%s at 0x%x: %s>" %\
          (self.__class__.__name__,id(self),vars(self))


class ConfigObj(OptionParser):
    """custonized OptionParser for getting options from config file"""

    def __init__(self,defaults,groups={},usage=None,description=None,version=None,config='--config'):
        """
    
        :param config: config switch name
        """
        self._configSwitch = config
        self._groups = groups
        OptionParser.__init__(self,usage=usage,description=description,version=version,conflict_handler="resolve")
        self._initialize(defaults)
        
    def _initialize(self,defaults):
        """put in default from formatted list

        :param defaults: default config options
        :rtype: None
        """
        groups = {}
                
        for n,id,d,t,h,g in defaults:
            # group option
            if g and g not in groups:
                groups[g] = OptionGroup(self,g,self._groups.get(g,"")) 

            action = "store_true" if t == "bool" else "store"
            if t == 'bool':
                if not g: 
                    self.add_option(n,dest=id,action="store_true",default=d,help=h)
                else:
                    groups[g].add_option(n,dest=id,action="store_true",default=d,help=h)
            else:
                if not g: 
                    self.add_option(n,dest=id,default=d,type=t,help=h)
                else:
                    groups[g].add_option(n,dest=id,default=d,type=t,help=h)

        # add group 
        for _,group_obj in groups.items():
            self.add_option_group(group_obj)

                                
    def _read_yaml(self,path):
        """Opens the input yaml file and returns parsed contents

        :param path: path of yaml file
        :type path: str
        :rtype: dict
        :raises: ErrrorReadingConfig, ErrorParsingConfig
        """
        try:
            
            with open(path) as my_config:
                config_values =  yaml.safe_load(my_config)
                unknown = [v for v in config_values if v not in self.defaults]

                if unknown:
                    logging.warning('unknown config file values (ignoring): %s' % ','.join(unknown))
                    config_values = {k:v for k,v in config_values.items() if k not in unknown}
                    
                return config_values
                    
        except IOError,e:
            raise ErrorReadingConfig('cannot find config file=%s' % path)
        
        except YAMLError,e:
            raise ErrorParsingConfig('error parsing content of config=%s' % path)

        except Exception,e:
            raise ErrorReadingConfig('error processing config file=%s' % e)


    def parse_args(self,args=None):
        """parsing the command-line arguments, rewritten for config files
        
        :rtype: tuple 
        """
        if args and isinstance(args,basestring): args = args.split()
        rargs = self._get_args(args)
                
        if self._configSwitch in rargs:
            config_index = rargs.index(self._configSwitch)
            config_file = rargs[config_index+1]
            del rargs[config_index]
            del rargs[config_index]
            file_config = self._read_yaml(config_file) 
            self.defaults = dict(self.defaults.items()+file_config.items()) 

        config,_ = OptionParser.parse_args(self,args=rargs)
        return config

    def parse_known_args(self,args=None):
        """Parse only known arguments

        Note : assumes only long arguments 

        :param args: the argv input
        """
        if args and isinstance(args,basestring): args = args.split()
        rargs = [o for o in self._get_args(args) if o.split('=')[0] in self._long_opt]
        config,_ = OptionParser.parse_args(self,args=rargs)
        return config
    
    def default_config(self):
        """returns a config with default values"""
        return self.get_default_values()

    def get_default_values(self):
        return ConfigAttrs(self.defaults)


if __name__ == "__main__":

    my_options = [("-s","--start","start",0,"int","start name"),
                  ("-r","--readme","readme",'',"str","readme message"),
                  ("-b","--b","b",True,"int","boolea switch")]
    
    z = ConfigObj(my_options)
    options,other = z.parse_args()
