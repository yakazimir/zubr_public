# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

from nose.tools import assert_equal,raises,assert_not_equal,ok_
from zubr.Aligner import Alignment,Decoding,Aligner
import numpy as np
import logging

D = True 
try: 
    import dill
except ImportError:
    D = False
    
class TestAlignment(object):
    pass 


class TestModel(object):

    def __init__(self):
        self.empty_aligner1 = Aligner("IBM1")
        self.empty_aligner2 = Aligner("IBM2") 

    def test_picklable1(self):
        if D: 
            assert_equal(dill.pickles(self.empty_aligner1),True)

    def test_picklable2(self):
        if D: 
            assert_equal(dill.pickles(self.empty_aligner2),True)

    # train model1 on small (500 pairs) of hansards corpus
    # def test_train_hansards_model1(self):
    #     aligner = Aligner("IBM1")
    #     gen_config.atraining = 'examples/hansards'
    #     gen_config.amax = 120
    #     gen_config.aiters = 3
    #     aligner.train(config=gen_config)

    # def test_train_hansards_model2(self):
    #     aligner = Aligner("IBM2")
    #     gen_config.atraining = 'examples/hansards'
    #     gen_config.amax = 120
    #     gen_config.aiters = 2
    #     aligner.train(config=gen_config) 
        
class TestDecoding(object):

    def test_equality(self):
        d1 = Decoding(1)
        d2 = Decoding(1)
        assert_equal(d1,d2)
        assert_equal(d1._ts_array,d2._ts_array) 

    def test_not_equal(self):
        d1 = Decoding(1)
        d2 = Decoding(2)
        assert_not_equal(d1,d2)
        ok_(d1 != d2)

    def test_mult_prob(self):
        d1 = Decoding(1)
        d1.prob *= 0.4
        assert_equal(d1.prob,0.4)

    def test_pickable(self):
        if D: 
            assert_equal(dill.pickles(Decoding(1)),True)

class TestAlignment(object):

    def __init__(self):
        pass 

    ## testing that the kbest returns correct output

    def test_k_best_3(self):
        a = Alignment.make_empty(3,3)
        record = a.problist
        ml = a.ml
        record[0][0] = 0.1;record[0][1] = 0.3; record[0][2] = 0.6
        record[1][0] = 0.2;record[1][1] = 0.3; record[1][2] = 0.3
        record[2][0] = 0.2;record[2][1] = 0.4; record[2][2] = 0.6
        ml[0] = 2;ml[1] = 2; ml[2] = 2
        kbest_list = a.alignments(50)
        assert_equal(len(kbest_list),27)

    # def test_k_best_4(self):
    #     a = Alignment.make_empty(4,4)
    #     record = a.problist
    #     ml = a.ml
    #     record[0][0] = 0.1;record[0][1] = 0.3; record[0][2] = 0.6; record[0][3] = 0.3
    #     record[1][0] = 0.2;record[1][1] = 0.3; record[1][2] = 0.3; record[1][3] = 0.1
    #     record[2][0] = 0.2;record[2][1] = 0.4; record[2][2] = 0.6; record[2][3] = 0.4
    #     record[3][0] = 0.2;record[3][1] = 0.4; record[3][2] = 0.6; record[3][3] = 0.15
    #     ml[0] = 2;ml[1] = 2; ml[2] = 2; ml[3] = 2
    #     align = a.alignments(500) 
    #     assert_equal(len(align),256)
