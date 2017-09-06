# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

utilities for building Zubr graph objects

"""
import codecs
import numpy as np
from collections import defaultdict

__all__ = [
    "str_weighted_graph",
    "weighted_graph",
]


def _read_weighted(string_rep):
    """The main function for parsing lines 

    :param string_rep: the string rep 
    :raises: ValueError 
    """
    edge_list  = []
    span_list  = []
    weights    = []
    nodes      = set()
    curr_node  = None
    curr_start = None
    start_ends = defaultdict(set)

    for k,entry in enumerate(string_rep):
        entry = entry.strip()

        ## tries to parse lines 
        try:
            start,end,weight = entry.split('\t')
            start = int(start); end = int(end); weight = float(weight)
            #identifier = len(start_ends[(start,end)])
            #start_ends[(start,end)].add((start,end,weight))

            edge_list.append([start,end])
            weights.append(weight)

            ## first node 
            if curr_node is None:
                curr_node = start
                curr_start = k

            ## next node 
            elif curr_node != start:
                span_list.append([curr_start,k-1])
                curr_node = start
                curr_start = k

            ## add nodes 
            nodes.add(start)
            nodes.add(end)
            
        except Exception,e:
            raise ValueError('Error reading graph line: %s' % entry)

    ## the final edges
    span_list.append([curr_start,k])
        
    assert len(weights) == len(edge_list),"Graph parser error: weights"
    assert len(span_list) == len(nodes)-1,"Graph parser error: span list"
    
    edge_list_numpy = np.array([x[-1] for x in edge_list],dtype=np.int32)
    span_list_numpy = np.array(span_list,dtype=np.int32)
    weight_numpy = np.array(weights,dtype='d')

    return (edge_list_numpy,span_list_numpy,weight_numpy,len(nodes))

def str_weighted_graph(string_input):
    """Build a weighted graph from a string representation 

    :param string_input: the input graph to parse
    :type string_input: basestring 
    """
    lp = [l.strip() for l in string_input.split('\n') if l.strip()]
    ## numpy representations 
    return _read_weighted(lp)

def weighted_graph(path):
    """Build a weighted graph from file 

    :param path: the path to the graph file 
    """
    lp = []
    
    with codecs.open(path,encoding='utf-8') as my_graph:
        for line in my_graph:
            line = line.strip()
            if not line: continue
            lp.append(line)

    ## numpy representations 
    return _read_weighted(lp)
