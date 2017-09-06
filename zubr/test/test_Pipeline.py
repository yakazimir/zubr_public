# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

from nose.tools import assert_equal,raises,assert_not_equal,ok_
from zubr.Pipeline import Pipeline,PipelineError,_set_logger as setl
from zubr.Pipeline import argparser as _generic_pipeline_config
#from zubr import gen_pconfig

D = True 
try: 
    import dill
except ImportError:
    D = False

gen_pconfig = _generic_pipeline_config().get_default_values() 

def routine1(config):
    return 10

def routine2(config):
    return 20

class TestPipeline(object):

    ## test logger level 
    
    @raises(PipelineError)
    def test_logger_unknown_level(self):
        gen_pconfig.loglevel = 'UNKNOWN' 
        setl(gen_pconfig)

    @raises(PipelineError)
    def test_notcallable(self):
        p = Pipeline(gen_pconfig,["str"])
        p()
        
    @raises(TypeError)
    def test_raises_typerror(self):
        p = Pipeline()

    def test_logger_level1(self):
        gen_pconfig.loglevel = 'DEBUG'
        setl(gen_pconfig)
        
    def test_logger_level2(self):
        gen_pconfig.loglevel = 'debug'
        setl(gen_pconfig)

    def test_logger_level3(self):
        gen_pconfig.loglevel = 'INFO'
        setl(gen_pconfig)

    def test_logger_level4(self):
        gen_pconfig.loglevel = 'CRITICAL'
        setl(gen_pconfig)

    def test_logger_level5(self):
        gen_pconfig.loglevel = 'WARNING'
        setl(gen_pconfig)

    def test_logger_level6(self):
        gen_pconfig.loglevel = 'notset'
        setl(gen_pconfig)

    ## test start/stop with example pipeline

    def test_empty_pipeline(self):
        gen_pconfig.loglevel = 'notset'
        p = Pipeline(gen_pconfig,[])
        ok_(p.start == 0)
        ok_(p.end == 1000)

    def test_nonempty_pipeline(self):
        tasks = [routine1,routine2]
        p = Pipeline(gen_pconfig,tasks)
        p()
        assert_equal(p.level,2)
        assert_equal(len(p.history),2)

    def test_nonempty_pipeline2(self):
        gen_pconfig.start = 1
        tasks = [routine1,routine2]
        p = Pipeline(gen_pconfig,tasks)
        assert_equal(p.level,1) 
        p.run_pipeline()
        assert_equal(p.level,2)
        assert_equal(p.history[0][-1],"skipped")

    def test_nonempty_pipeline3(self):
        gen_pconfig.start = 0
        tasks = [routine1,routine2]
        p = Pipeline(gen_pconfig,tasks)
        assert_equal(p.level,0) 
        p()
        assert_equal(p.level,2)

    def test_nonempty_pipeline4(self):
        gen_pconfig.start = 0
        gen_pconfig.end = 1
        tasks = [routine1,routine2]
        p = Pipeline(gen_pconfig,tasks)
        assert_equal(p.end,1) 
        p()
        assert_equal(p.level,1)
        assert_equal(len(p.history),1)

    def test_report_str(self):
        p = Pipeline(gen_pconfig,[])
        ok_(str(p))
        
    ## check that pipeline can be pickled

    def test_picklable(self):
        if D:
            ok_(dill.pickles(Pipeline(gen_pconfig,[])))

    ## level setter

    def test_setlevel(self):
        p = Pipeline(gen_pconfig,[])
        p.level = 2
        
    
    

    
