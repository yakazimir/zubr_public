#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""
import numpy as np
cimport numpy as np

def apply_binary_sort(random_array):
    """python method for accessing binary_insert_sort

    :param random_array: a random array to be sorted
    :type random_array: list of np.array
    """
    if isinstance(random_array,list):
        random_array = np.array(random_array,dtype='d')

    alen = random_array.shape[0]
    sort = np.ndarray(alen,dtype=np.int32)
    sort.fill(-1)
    for i in range(alen):
        val = random_array[i]
        binary_insert_sort(i,val,random_array,sort)

    return ([i for i in sort],[random_array[i] for i in sort])

cdef int binary_insert_sort(int index,double score, double[:] problist,int[:] sort) except -1:
    """insert an item into a sorted list using a binary-search 

    :param index: current index under consideration
    :param score: current score of index
    :param problist: probabilities associated with indices
    :param sort: sorted list containing indices of items    
    """
    cdef int mid,start,end
    cdef double sprob
    cdef int loops = 0
        
    if index == 0:
        sort[0] = index

    elif score > (<double>problist[sort[0]]):
        sort[1:] = sort[0:-1]
        sort[0] = index

    elif score < (<double>problist[sort[index-1]]):
        sort[index] = index

    else:
        start = 0
        end = index - 1

        while True:

            loops += 1
            mid = (start+end)/2
            sprob = <double>problist[sort[mid]]

            if mid == start == end:
                sort[mid+1:] = sort[mid:-1]
                sort[mid] = index
                break

            if score == sprob:
                sort[mid+1:] = sort[mid:-1]
                sort[mid] = index
                break

            if (score > sprob):
                end = mid

            elif (score < sprob):
                start = (mid + 1)

