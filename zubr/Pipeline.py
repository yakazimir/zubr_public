#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""

from __future__ import print_function
import os
import sys
import traceback
import logging
import datetime
from zubr.util.loader import load_script,load_module
from zubr.util import ConfigObj
from zubr.util.os_util import make_experiment_directory,print_readme
from zubr.util.pipeline_util import make_run_script,make_restore
import time

__all__ = [
    "PipelineError",
    "Pipeline",
    "pipeline_params"
]

class PipelineError(Exception):
    pass

class Pipeline(object):
    """class for defining and running experiment pipelines"""

    def __init__(self,config,tasks):
        """

        :param config: global experiment configuration
        :param tasks: tasks to be completed
        :param _history: record of tasks completed
        :param start: place/level to start the pipeline
        :param end: place/level to end the pipeline
        """
        self.config = config
        self.tasks = tasks
        self.dir = config.dir
        self._history = []
        self.start = self.config.start
        self.end   = self.config.end
        self._level = self.config.start
        self._logger.debug('initialized pipeline, starting at: %s' % self.start)
        try: 
            self._skips = [] if not config.skip_tasks or not config.skip_tasks.strip() else\
            [int(i) for i in config.skip_tasks.split("+")]
        except ValueError:
            self._skips = []

    @property
    def history(self):
        """returns the current execution history

        :rtype: list
        """
        return self._history

    @property
    def level(self):
        """return the current processing level

        :rtype: int
        """
        return self._level

    @level.setter
    def level(self,new):
        """reset the current pipeline (processing) level

        :rtype: None
        """
        self._level = new
        self._logger.info('changed pipeline level: %d' % new)

    @property
    def _logger(self):
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level) 

    @staticmethod
    def _getname(task):
        """returns the full name of task

        :rtype: str
        """
        return task.__module__+"."+task.__name__
    
    def report(self):
        """give a report of the pipeline

        :returns: string with information about each executed task
        :rtype: str
        """
        result = 'run: %s\n' % str(datetime.datetime.now())
        for (l,(name,time)) in enumerate(self.history):
            result += "level: %s, task: %s, time (seconds): %s\n" % (l,name,str(time))
        return result
        
    def run_pipeline(self):
        """execute the list of tasks begining with start point

        :returns: None
        """
        ## log the pipeline information
        for l,task in enumerate(self.tasks):

            if not hasattr(task,'__call__'):
                raise PipelineError('task is not callable: %s' % str(task))

            task_name = type(self)._getname(task)

            if l < self.start or l in self._skips:
                self._logger.info('skipping task: %s,level=%d' % (task_name,self._level))
                self._history.append((task_name,"skipped")) 
                continue

            elif l >= self.end:
                break

            start_time = time.time()
            task(self.config)
            self._history.append((task_name,time.time()-start_time))
            self._logger.info('finished task: %s,level=%d' % (task_name,self._level))
            if self._level < self.end:
                self._level += 1

    __call__ = run_pipeline
            
    def dump(self):
        """dump the current state of the pipeline instance

        :param path: path where to dump pickle object (e.g. working dir)
        :type path: str 
        :rtype: None
        """
        import pickle
        out = os.path.join(self.dir,"pipeline.p") 
        with open(out,'wb') as backup:
            pickle.dump(self,backup)

    @classmethod
    def restore(cls,path):
        """restore a partially run pipeline

        :param path: example working directory path
        :type path: str 
        :rtype: Pipeline
        """
        import pickle
        out = os.path.join(path,"pipeline.p")
        with open(out,'rb') as pipeline:
            old =  pickle.load(pipeline)
            old.start = old._level
            old._history = []
            old._logger.debug('restoring at level: %d' % old.start)
            return old

    def print_report(self):
        """prinout a pipeline report

        :param out: file to print to
        :type out: str
        :rtype: None 
        """
        b = '--------'*3
        out = os.path.join(self.dir,"report.txt") 
        with open(out,'a') as report:
            print (b,file=report)
            print (self,file=report)

    def __str__(self):
        return self.report()

    
##########################################
########################################## CLI
##########################################

def pipeline_params():
    """main parameters for running a zubr pipeline or experiment

    :rtype: tuple
    :returns: description of option types with name, list of options 
    """
    options = [
        ("--start","start",0,int,
         "start place in pipeline [default=0]","Pipeline"),
        ("--end","end",1000,int,
         "end place in pipeline [default=0]","Pipeline"),
        ("--dir","dir",'',"str",
         "output directory [default='']","Pipeline"),
        ("--override","override",False,'bool',
         "override existing dir [default=False]","Pipeline"),
        ("--readme","readme",'','str',
         "readme message [default='']","Pipeline"),
        ("--backup","backup",False,'bool',
         "back up everything [default=False]","Pipeline"),
        ("--encoding","encoding",'utf-8','str',
         "default encoding [default=utf-8]","Pipeline"),
        ("--lowercase","lowercase",True,'bool',
         "lowercase all data [default=True]","Pipeline"),
        ("--log","log",'','str',
         "log to file [default='']","Pipeline"),
        ("--exitonfail","exitonfail",True,'bool',
         "exit on system fail [default=True]","Pipeline"),
        ("--logfile","logfile",'pipeline.log','str',
         "log to file [default='pipeline.log']","Pipeline"),
        ("--loglevel","loglevel",'DEBUG','str',
         "log level [default='DEBUG']","Pipeline"),
        ("--dump_models","dump_models",False,'bool',
         "backup the models when trained/failed [default=False]","Pipeline"),
        ("--cleanup","cleanup",True,'bool',
         "cleanup when possible [default=True]","Pipeline"),
        ("--build_data_only","build_data_only",False,"bool",
        "build the data and exit [default=False]","Pipeline"),
        ("--skip_tasks","skip_tasks","","str",
        "Tasks to skip [default='']","Pipeline"),
        ("--pipeline_backup","pipeline_backup",False,"bool",
        "Use the pipeline backup protocol [default=False]","Pipeline"),
        ("--restore_pipeline","restore_pipeline",'',"str",
        "Restore pipeline to a certin level [default='']","Pipeline"),        
    ]

    pipeline_group = {"Pipeline":"Pipeline settings and defaults"}
    return (pipeline_group,options)

def argparser():
    """return an pipeline argument parser using defaults

    :rtype: zubr.util.config.ConfigObj
    :returns: default argument parser
    """
    from zubr import _heading
    from _version import __version__ as v
    from zubr.util import ConfigObj
    
    usage = """python -m zubr pipeline [options]"""
    d,options = pipeline_params()
    argparser = ConfigObj(options,d,usage=usage,description=_heading,version=v)
    return argparser 


def _set_logger(config):
    """setup the pipeline logger
    
    :param config: pipeline configuration object
    :type config: zubr.util.Config.ConfigAttrs
    :rtype: None
    :raises: PipelineError
    """
    level = config.loglevel.upper()
    path = config.logfile
    if not hasattr(logging,level):
        raise PipelineError('logging level not known: %s' % level)

    level = getattr(logging,level)
    if path == 'None' or path == 'stdout':
        logging.basicConfig(level=level)
    else:
        path = os.path.join(config.dir,path)
        logging.basicConfig(filename=path,level=level)

def main(argv):
    """main execution method for starting a pipeline

    :param argv: user input 
    :type argv: list
    :rtype: None
    """
    
    from zubr import _heading as h
    from _version import __version__ as v

    if len(argv) == 1:
        raise PipelineError('must specify pipeline script...')

    perror = False
    #logging.basicConfig(level=logging.DEBUG)
    usage = """python -m zubr pipeline script [options]"""
    pipelined,options = pipeline_params() 
    tasks = []
    params = options
    descriptions = pipelined

    ## parse pipeline script and get list of modules
    try:
        ## running a script 
        pipeline_script = load_script(argv[1])
        try: 
            script_tasks = pipeline_script.tasks
            script_params = pipeline_script.params
            descriptions.update(pipeline_script.description)
            params += script_params

        except AttributeError,e:
            raise PipelineError('missing script attr: %s' % e)
        
        # Iterate each processing task
        for task in script_tasks:
            
            if task in vars(pipeline_script):
                tasks.append(vars(pipeline_script)[task])
            else:
                mod = load_module(task)
                if hasattr(mod,"params"):
                    d,p = mod.params()
                    descriptions.update(d)
                    params += p

                ## change this to take either mod.main or
                tasks.append(mod.main)

        ## extra params to add to 
        if hasattr(pipeline_script,"ex_params"):
            for extra in pipeline_script.ex_params:
                mod = load_module(extra)
                if hasattr(mod,"params"):
                    d,p = mod.params()
                    descriptions.update(d)
                    params += p

        argparser = ConfigObj(params,descriptions,usage=usage,description=h,version=v)
        config = argparser.parse_args(argv[2:])

        ## logging
        #logging.basicConfig(filename=logFile,level=logging.DEBUG)          

        if not config.dir:
            raise PipelineError('must specify working directory..')

        if os.path.isdir(config.dir) and config.start > 0:
            desired_start = config.start
            config.restore_old(config.dir)
            config.start = desired_start
            config.restore = None
            task_pipeline = Pipeline(config,tasks)

        elif os.path.isdir(config.dir) and config.start == -1:
            task_pipeline = Pipeline.restore(config.dir) 
            
        else:
            working_dir = make_experiment_directory(config.dir,config)
            config.dir = os.path.abspath(working_dir)
            if config.readme:
                print_readme(config.readme,config.dir)
            task_pipeline = Pipeline(config,tasks)

        ## setup logger
        _set_logger(config)
        task_pipeline._logger.info('input: %s' % ' '.join(argv))
        
        make_run_script(config,' '.join(argv))
        ## run pipeline
        task_pipeline()

    #except KeyError,e:
    except Exception,e:
        perror = True
        traceback.print_exc(file=sys.stdout)
    finally:

        ## try to back up what's been done already
        try: 
            config.print_to_yaml(config.dir)
            task_pipeline.print_report()

            ## check if the pipeline finished 
            #if task_pipeline.level != task_pipeline.end or perror:
            if task_pipeline.level < len(task_pipeline.tasks):
                make_restore(config,argv,task_pipeline.level)
            
        except (NameError,IOError):
            pass

        
        
if __name__ == "__main__":
    main(sys.argv[1:])
