# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

from nose.tools import assert_equal,ok_,raises,assert_not_equal
from zubr.Dataset import RankDataset,EMPTY_RANK
import numpy as np

d1 = np.array([
    np.array([0,1,3]),
    np.array([3,4,5]),
    np.array([4, 7, 9])],
    dtype='object')

d1_orig = np.array([
    np.array([u"this is my sentence"]),
    np.array([u"another sentence"]),
    np.array([u"yet another one"])],
    dtype='object')

gold = np.array([0,1,0],dtype=np.int32)


class TestRankDataset(object):

    def __init__(self):
        self.ex1 = RankDataset(d1,d1_orig,gold)
    
    def test_empty(self):
        assert_equal(EMPTY_RANK.is_empty,True)

    def test_org_order(self):
        order = self.ex1._dataset_order.tolist()
        size_range = range(len(order))
        assert_equal(order,size_range)

    def test_shuffle(self):
        order1 = self.ex1._dataset_order.tolist()
        self.ex1.py_shuffle()
        order2 = self.ex1._dataset_order.tolist()
        assert_not_equal(order1,order2)

    def test_double_shuffle(self):
        self.ex1.py_shuffle()
        order1 = self.ex1._dataset_order.tolist()
        self.ex1.py_shuffle()
        order2 = self.ex1._dataset_order.tolist()
        assert_not_equal(order1,order2)

    def test_next_with_shuffle(self):
        self.ex1.py_shuffle()
        order1 = self.ex1._dataset_order.tolist()
        next_item = self.ex1.py_next()
        assert_equal(next_item.global_id,order1[0])
        nnext_item = self.ex1.py_next()
        assert_equal(nnext_item.global_id,order1[1])

    @raises(ValueError)
    def test_mismatch_size(self):
        RankDataset(d1,d1_orig,np.empty(0,))
    

