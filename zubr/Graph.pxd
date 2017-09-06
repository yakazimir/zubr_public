import sys
import numpy as np
cimport numpy as np
from ZubrClass cimport ZubrSerializable

cdef extern from "math.h":
    bint isnan(double)
    bint isinf(double)
    double log(double)

## adjacency class
cdef class Adjacency:
    pass 

cdef class DirectedAdj(Adjacency):
    cdef np.ndarray edges
    cdef np.ndarray node_spans

cdef class GraphBase(ZubrSerializable):
    pass

cdef class DirectedGraph(GraphBase):
    cdef DirectedAdj adj
    cdef int num_nodes
    cpdef np.ndarray shortest_paths(self,int source,int k=?,bint trace=?)

cdef class UnweightedDirectedGraph(DirectedGraph):
    pass

cdef class WeightedDirectedGraph(DirectedGraph):
    cdef np.ndarray weights

## weighted dag

cdef class WeightedDAG(WeightedDirectedGraph):
    pass

cdef class UnweightedDAG(UnweightedDirectedGraph):
    pass

cdef class WordGraph(UnweightedDAG):
    pass

## path classes

cdef class PathBuffer:
    cdef set seen
    cdef list sortedpaths
    cdef void push(self,double cost, DirectedPath path)
    cdef DirectedPath pop(self)
    cdef bint empty(self)

cdef class Path:
    cdef np.ndarray seq
    cdef int size

cdef class DirectedPath(Path):
    cdef double score
    cdef np.ndarray node_scores

cdef class KShortestPaths:
    cdef np.ndarray _A
    cdef int _k
