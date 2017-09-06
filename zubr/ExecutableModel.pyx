# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

Implementing executable models 

"""

import subprocess 
import traceback
import logging
import sys
import os
import signal
import re
import time
import fcntl
import errno
from zubr.ZubrClass cimport ZubrSerializable

cdef class ExecutableModel(ZubrSerializable):
    """Base class for executable models """

    def execute(self, einput):
        """Execute a given input and return raw output 

        :param einput: the input to execute 
        """
        raise NotImplementedError 
        
    @classmethod
    def from_config(cls,config):
        """Setup a model from a configuration 

        :param config: the main configuration
        """
        raise NotImplementedError

    def __enter__(self):
        ## enter 
        raise NotImplementedError 

    def __exit__(self, exc_type, exc_val, exc_tb):
        ## 
        raise NotImplementedError

class ProcTimeout(Exception):
    pass 
        
cdef class PromptingSubprocess(ExecutableModel):
    """An executable model that involves calling some subprocess that you can prompt

    -- The prompting subprocess has two parts: First, the subprocess to call, or self.start_up, 
    and the output pattern to look for, self.out_pat.

    """

    def __init__(self,start_up,timeout):
        """Create a promptingsubprocess instance

        :start_up: the start up sub process to get 
        :param timeout: the maximum timeout 
        """
        self.start_up = start_up
        self.timeout  = timeout 
        
        ## main subprocess, boilerplate subprocess stuff
        self.proc = subprocess.Popen(self.start_up,
                                         bufsize=0,
                                         shell=True,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         close_fds=True)

        fd = self.proc.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        
        ## get rid of initial
        try: 
            first_out = self.prompt(None)
        except ProcTimeout:
            time.sleep(1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def execute(self,einput):
        """The main execution function, used for evaluating 
        truth. 

        :param einput: the raw input to process 
        """
        raise NotImplementedError

    __call__ = prompt

    def prompt(self, einput):
        """Main method for executing giving an input

        :param einput: the input to execute
        """
        raise NotImplementedError

    exit = __del__

    def __del__(self):
        os.system("""pkill -f '%s'""" % self.start_up)
        ## this doesn't seem to work 
        #time.sleep(0.1)
        #self.proc.kill()
        #os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)

cdef class LispModel(ExecutableModel):
    """An executable mode that involves lisp"""
    pass

### EXAMPLE EXECUTORS

GEO_C = re.compile(r'\<match\=true\>|\<match\=false\>')
GEO_P = re.compile(r'A \= (.+)')


cdef class GeoExecutor(PromptingSubprocess):
    """An executor for the geoquery domain"""

    def execute(self,einput,functional=True):
        """Execute a given funql geoquery input to the background geobase

        :param einput: the input to execute 
        :rtype: bool 
        """
        einput = functionalize(einput) if functional else einput
        query = "execute_funql_query(%s,A)." % einput
        try: 
            return self.prompt(query)
        except Exception,e:
            self.logger.error(e,exc_info=True)
            return None

    def evaluate(self,hypothesis,gold,functional=True):
        """Compare if a hypothesis executed to a gold

        :param hypothesis: the hypothesis representation 
        :param gold: the gold representation 
        """
        hyp = functionalize(hypothesis) if functional else hypothesis
        gol = functionalize(gold) if functional else gold
        query = "compare(%s,%s)." % (hyp,gol)
        
        try:
            return self.prompt(query)
        except Exception,e:
            self.logger.error(e,exc_info=True)
            return None
        
    def prompt(self,einput):
        """Execute input geoquery queries

        :param einput: the input to execute
        :param timeout: the maximum time to wait (0.0 by default) 
        """
        if einput:
            self.proc.stdin.write(einput+"\n\n")
            self.proc.stdin.flush()

        start  = time.time()
        output = ''

        ## run until the desired pattern is found, or runtime error 
        while not self.timeout or (time.time() < (start + self.timeout)):

            try:
                raw_output = self.proc.stdout.read()
            except IOError,e:
                if e.errno == errno.EAGAIN:
                    continue
                raise IOError(e)

            ## keep retrieving output if necessary
            if raw_output:
                output += raw_output
            else:
                time.sleep(0.01)

            ## check resulting pattern 
            if re.search(GEO_C,output):
                if '=true' in output:
                    return True
                return False

            ## error pattern
            
            list_out = re.search(GEO_P,output) 
            if list_out:
                return list_out.groups()[0]

        # ## raise timeout error
        time.sleep(0.1)
        raise ProcTimeout('Exceeded timeout on %s' % einput)

    @classmethod
    def from_config(cls,config):
        """Load an geo executor instance from a configuration 

        :param config: the configuration with the associated settings 
        """
        starter = "%s --quiet -l %s" % (config.prolog_binary,config.geo_script)
        return cls(starter,config.timeout)
        

### taken from jacob andreas
##

ARITY_SEP = '@'
ARITY_STR = 's'
ARITY_ANY = '*'


def functionalize(mrl):
    """Convert geoquery representation into a functional representation 

    :param mrl: the input mrl in a flat linear form 
    """
    stack = []
    r = []
    tokens = list(reversed(mrl.split()))

    #print tokens

    while tokens:
      it = tokens.pop()
      #print it
      if ARITY_SEP not in it:
        token = it
        arity = ARITY_STR
        logging.warn('unrecognized token: %s', it)
      else:
        token, arity = it.rsplit(ARITY_SEP)
      if arity == ARITY_STR:
        arity = 0
        arity_str = True
      elif not (arity == ARITY_ANY):
        arity = int(arity)
        arity_str = False
      
      if arity == ARITY_ANY or arity > 0:
        r.append(token)
        r.append('(')
        stack.append(arity)
      else:
        assert arity == 0
        if arity_str:
          r.append("'%s'" % token.replace('_', ' '))
        else:
          r.append(token)
          #print r
        while stack:
          top = stack.pop()
          if top == ARITY_ANY and tokens:
            r.append(',')
            stack.append(ARITY_ANY)
            break
          elif top != ARITY_ANY and top > 1:
            r.append(',')
            stack.append(top - 1)
            break
          else:
            r.append(')')

        if not stack and tokens:
          return None

    if stack:
      return None

    r = ''.join(r)

    # nasty hacks to fix misplaced _
    if '(_' in r:
      return None
    if ',_' in r and not ('cityid' in r):
      return None
    if '_),_)' in r:
      return None

    return r


### FACTOR

EXECUTORS = {
    "geoquery" : GeoExecutor,
}

def ExecuteModel(config):
    """Factory method for returning an executable model

    :param config: the main configuration 
    :returns: executable model class 
    """
    emodel = EXECUTORS.get(config.exc_model,None)
    if not emodel:
        raise ValueError('Uknown type of executable model: %s' % str(config.exc_model))
    return emodel

### CLI STUFF

def argparser():
    """Returns a configuration for executable models using defaults 

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser    
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr executable_model [options]"""
    d,options = params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser

def params():
    """The main parameters for running the aligners and/or aligner experiments 


    :rtype: tuple 
    :returns: description of option types with name,list of options 
    """
    from zubr import lib_loc
    geo_script = os.path.join(lib_loc,"experiments/technical_documentation/other_data/geoquery/eval.pl")
    
    options = [
        ("--prolog_binary","prolog_binary","swipl","str",
         "Location of the prolog binary [default='swipl']","GeoQuery"),
        ("--geo_script","geo_script",geo_script,"str",
         "The script for loading geoquery [default='']","GeoQuery"),
        ("--timeout","timeout",0.1,"float",
         "The maximum time out [default='']","PromptingSubprocess"),
        ("--exc_model","exc_model",'geoquery',"str",
         "The type of executable model to use [default='']","ExecutableModel"),
    ]

    model_group = {
        "ExecutableModel"     : "General settings for executable models",
        "PromptingSubprocess" : "Settings for starting subprocesses",
        "GeoQuery"            :"Geo executor settings (if used)",
        "LispModel"           :"General settings for lisp executable model (if used)",
    }

    return (model_group,options)

        
def main(config):
    """The main execution point 

    """
    pass
