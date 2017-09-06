# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

from nose.tools import assert_equal,ok_,raises,assert_not_equal
from zubr.Alg import apply_binary_sort
import random

random.seed(42)
RANDOM_LIST_1 = [float(i) for i in range(1,100)]
RANDOM_LIST_2 = [float(i)/300.0 for i in range(1,300)]
random.shuffle(RANDOM_LIST_1)
random.shuffle(RANDOM_LIST_2)

class TestAlg(object):

    def test_binary_insert_sort1(self):
        order,sorted_list = apply_binary_sort(RANDOM_LIST_1)
        assert_equal(sorted_list,sorted(sorted_list,reverse=True))

    def test_binary_insert_sort2(self):
        order,sorted_list = apply_binary_sort(RANDOM_LIST_2)
        assert_equal(sorted_list,sorted(sorted_list,reverse=True))
    

