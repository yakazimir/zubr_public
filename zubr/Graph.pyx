# cython: profile=True
# -*- coding: utf-8 -*-
# filename: Graph.pyx

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson


Graph implementations (for now, types of directed graphs), 
and associated  algorithms (shortest path, ...) 

"""
import time 
import sys
import gzip
import os
import cPickle as pickle
import logging
import numpy as np
cimport numpy as np
from collections import defaultdict
from heapq import heappush, heappop
from ZubrClass cimport ZubrSerializable
from zubr.util.graph_util import *
from cython cimport wraparound,boundscheck,cdivision
from libc.stdlib cimport malloc, free

cdef class GraphBase(ZubrSerializable):
    """Base class for graph classes """

    ## class methods for constructing graphs

    @classmethod
    def from_str(cls,string_rep):
        """Construct a graph from a string representation 

        -- different graph algorithms should implement this according to 
        constraints on the type of graphs they implement. 

        :param string_rep: the input representation of the graph
        :type string_rep: basestring 
        :returns: graph instance 
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls,file_path):
        """Construct a graph from a string representation in file

        -- different graph algorithms should implement this according to 
        constraints on the type of graphs they implement. 
        
        :param file_path: the path to the target file 
        :type file_path: basestring 
        :returns: graph instance 
        """
        raise NotImplementedError

    @classmethod
    def from_adj(cls):
        """Construct a graph from adjacency information

        :returns: graph instance 
        """
        raise NotImplementedError

    ## backup implementation

    def backup(self,wdir):
        """Back up a graph instance 

        :param wdir: the working directory 
        :rtype: None 
        """
        raise NotImplementedError
        
    @classmethod
    def load_backup(cls,config):
        """Load a backup 

        :params config: the main configuration 
        :returns: Graph instance 
        """
        raise NotImplementedError

    ## properties
    
    property number_nodes:
        """Attributes related to the number of nodes in graph"""
        
        def __get__(self):
            """Returns the number of nodes 
            
            :rtype: int  
            """
            raise NotImplementedError

    property is_directed:
        """Determine if the graph is directed"""

        def __get__(self):
            """Returns true if directed false otherwise 

            :rtype: bool
            """
            raise NotImplementedError 


    property is_weighted:
        """Determine if the graph has weights associated with it"""

        def __get__(self):
            """Returns true if weighted false otherwise 

            :rtype: bool
            """
            raise NotImplementedError

cdef class DirectedGraph(GraphBase):

    """Base class for directed graph types"""

    cpdef np.ndarray shortest_paths(self,int source,int k=1,bint trace=False):
        """Find shortest path(s) in a directed graph

        :param source: the source or starting node in the graph 
        :param k: the number of paths to return (default=1) 
        """
        raise NotImplementedError

    property is_directed:
        """Determine if the graph is directed"""

        def __get__(self):
            """Returns true if directed false otherwise 

            :rtype: bool
            :returns: True
            """
            return True

cdef class UnweightedDirectedGraph(DirectedGraph):
    """Base class for unweighted graphs"""

    def __init__(self,DirectedAdj adj,int num_nodes):
        """Constructs a weighted directed graph instance

        :param adj: the adjacency information 
        :param weights: the edge weights 
        :param num_nodes: the total number of nodes 
        """
        self.adj       = <DirectedAdj>adj
        self.num_nodes = num_nodes

    property is_weighted:
        """Determine if the graph has weights associated with it"""

        def __get__(self):
            """Returns true if weighted false otherwise 

            :rtype: bool
            :returns: False 
            """
            return False

    @classmethod
    def from_adj(cls,edges,span,size):
        """Construct a graph from adjacency information

        :param edges: the sparse edge representation 
        :param span: the span of nodes in edge list 
        :returns: graph instance 
        """
        adj = DirectedAdj(edges,span)
        return cls(adj,size)

    def __reduce__(self):
        ## pickle implementation
        return self.__class__,(self.adj,self.num_nodes)

    ## backup implementation

    def backup(self,wdir):
        """Back up a graph instance 

        :param wdir: the working directory 
        :rtype: None 
        """
        cdef DirectedAdj adj = <DirectedAdj>self.adj
        cdef np.ndarray edges,node_spans

        stime = time.time()
        gdir = os.path.join(wdir,"graph")
        if os.path.isdir(gdir):
            self.logger.info('Already backed up, skipping...')
            return 

        os.mkdir(gdir)
        ## save all the graph parts 
        edges = adj.edges
        node_spans = adj.node_spans
        fout = os.path.join(gdir,"graph_parts")
        np.savez_compressed(fout,edges,node_spans)

        ## save number of nodes
        info_file = os.path.join(gdir,"graph_info.p")
        with open(info_file,'wb') as my_info:
            pickle.dump({"size":self.num_nodes},my_info)

        self.logger.info('Backed up in %s seconds' % str(time.time()-stime))
                
    @classmethod
    def load_backup(cls,config):
        """Load a backup 

        :params config: the main configuration 
        :returns: Graph instance 
        """
        gdir = os.path.join(config.dir,"graph")
        stime = time.time()

        gfile = os.path.join(gdir,"graph_parts.npz")
        archive = np.load(gfile)
        edges = archive["arr_0"]
        spans = archive["arr_1"]
        ## create adjecency instance 
        adj = DirectedAdj(edges,spans)

        ginfo = os.path.join(gdir,"graph_info.p")
        with open(ginfo,'rb') as pd:
            info = pickle.load(pd)
            size = info["size"]
                                        
        instance = cls(adj,size)
        instance.logger.info('Loaded backup in %s seconds' % str(time.time()-stime))
        return instance
    

cdef class WeightedDirectedGraph(DirectedGraph):
    """Base class for weighted graphs"""

    def __init__(self,DirectedAdj adj,np.ndarray weights,int num_nodes):
        """Constructs a weighted directed graph instance

        :param adj: the adjacency information 
        :param weights: the edge weights 
        :param num_nodes: the total number of nodes 
        """
        self.adj       = <DirectedAdj>adj
        self.weights   = weights
        self.num_nodes = num_nodes
        
    property is_weighted:
        """Determine if the graph has weights associated with it"""

        def __get__(self):
            """Returns true if weighted false otherwise 

            :rtype: bool
            :returns: True
            """
            return True

    @classmethod
    def from_adj(cls,edges,span,size,weights):
        """Construct a graph from adjacency information

        :param edges: the sparse edge representation 
        :param span: the span of nodes in edge list 
        :param weights: the weight edge vector
        :returns: graph instance 
        """
        adj = DirectedAdj(edges,span)
        return cls(adj,weights,size)

    @classmethod
    def from_str(cls,string_input):
        """Construct a weighted directed graph from a string representation 

        Below is an example input (start node, end node, weight, each tab delimited): 

        ex2 =\
        0	1	5
        0	2	3
        1	2	3
        1	3	6
        2	3	7
        2	4	4
        2	5	2
        3	4	-1
        3	5	3
        4	5	-2

        A few rules about the graph formatting: 

           --- Assumes graph node numbers form a toplogical sort 
           (if you have a graph, we have some code for making a topological sort, see wrapper/foma)
           -- There should only be one terminating state

        :param string_rep: the input representation of the graph 
        :type string_input: basestring 
        :returns: weighted directed graph instance 
        """
        edges,spans,weights,v = str_weighted_graph(string_input)
        adj = DirectedAdj(edges,spans)

        return cls(adj,weights,v)
    
    @classmethod
    def from_file(cls,file_path):
        """Construct a graph from a string representation in file

        -- different graph algorithms should implement this according to 
        constraints on the type of graphs they implement. 
        
        :param file_path: the path to the target file 
        :type file_path: basestring 
        :returns: graph instance 
        """
        edges,spans,weights,v = weighted_graph(file_path)
        adj = DirectedAdj(edges,spans)

        return cls(adj,weights,v)

    def __reduce__(self):
        ## pickle implementation
        return self.__class__,(self.adj,self.weights,self.num_nodes)

## instantiable classes

cdef class UnweightedDAG(UnweightedDirectedGraph):

    @boundscheck(False)
    cpdef np.ndarray shortest_paths(self,int source,int k=1,bint trace=False):
        """Find shortest path(s) in a directed graph

        :param source: the source or starting node in the graph 
        :param k: the number of paths to return (default=1) 
        """
        cdef DirectedAdj adj = self.adj
        cdef int v = self.num_nodes

        ## generate weights on the fly 
        cdef double[:] weights = np.zeros((v,),dtype='d')
        cdef KShortestPaths shortest
        cdef np.ndarray paths

        if trace: ptime = time.time()
        # ## run the k best paths
        shortest = dag_shortest_paths(adj,weights,k,v)
        paths = shortest.paths
        if trace:
            self.logger.info('Finished finding %d paths in: %s seconds' %\
                                 (paths.size,str(time.time()-ptime)))
        return paths

cdef class WeightedDAG(WeightedDirectedGraph):
    """Implementation of a weighted acyclic directed graph (DAG)

    Note : shortest path algorithm here uses linear time algorithm 
    based on taking a toplogical sort of the graph. This allows for 
    negative weights on edges, which is often needed for applications
    in zubr. 

    """

    @boundscheck(False)
    cpdef np.ndarray shortest_paths(self,int source,int k=1,bint trace=False):
        """Find shortest path(s) in a directed graph

        :param source: the source or starting node in the graph 
        :param k: the number of paths to return (default=1) 
        """
        cdef DirectedAdj adj = self.adj
        cdef int v = self.num_nodes
        cdef double[:] weights = self.weights
        cdef KShortestPaths shortest
        cdef np.ndarray paths

        if trace: ptime = time.time()
        # ## run the k best paths
        shortest = dag_shortest_paths(adj,weights,k,v)
        paths = shortest.paths
        if trace:
            self.logger.info('Finished finding %d paths in: %s seconds' %\
                                 (paths.size,str(time.time()-ptime)))
        return paths

cdef class WordGraph(UnweightedDAG):
    
    """Implementation of a word graph for machine translation decoding, which 
    makes certain adjustment to how the shortest paths are computed
    """
    pass


## adjacency class

cdef class Adjacency:
    """Base class for adjacency types"""
    pass 

cdef class DirectedAdj(Adjacency):
    """Representing adjacency information for a directed graph

    The adjacencies are stored using a sparse matrix representation: 

    edge list, a two dimensional  : [ [ dest identifier ]<1> .. [ dest identifier ]<2> ... ] 
    node span :  [ [ start end ]<i> [ start end ]<i'> .. ] 
    """

    def __init__(self,edge_list,node_spans):
        """Creates an adj (or adjancecy) object 

        :param edge_list: the list of directed edges 
        :type edge_list: np.ndarray 
        :param edge_span: the span associated with each node in edge_list
        :type edge_span: np.ndarray 
        """
        self.edges      = edge_list
        self.node_spans = node_spans

    def __reduce__(self):
        return DirectedAdj,(self.edges,self.node_spans)

## path class

cdef class DirectedPath(Path):
    """Class for representation a directed path """

    def __init__(self,np.ndarray path,np.ndarray[ndim=1,dtype=np.float_t] nscores,
                     double score,int size):
        """Initializes a path instance 

        :param path: the path 
        :param score: the path score 
        """
        self.seq   = path
        self.score = score
        self.size  = size
        self.node_scores = nscores

    property invalid:
        """Path if the path is valid (i.e., final node has a normal (non-inf) score)"""

        def __get__(self):
            """Returns whether teh path is valid 

            :rtype: bool 
            """
            cdef double[:] ns = self.node_scores
            
            return isinf(ns[-1])

    property path:
        """The underlying path sequence"""

        def __get__(self):
            """Return the raw, numpy sequence information 

            :rtype: np.ndarray
            """
            cdef np.ndarray seq = self.seq
            cdef double cost = self.score
            return (seq,cost)


EMPTY_PATH = DirectedPath(np.array([-1],dtype=np.int32),
                              np.array([np.inf],dtype='d'),
                              np.inf,0)

## path buffer (working off of the networkx implementation here )
## http://networkx.readthedocs.io/en/stable/_modules/networkx/algorithms/simple_paths.html#shortest_simple_paths

cdef class PathBuffer:
    """Cython level path buffer to keep track of candidate paths (e.g., when computing k-best paths) """

    def __init__(self):
        """Initialized a path buffer instance 

        """
        self.seen = set()
        self.sortedpaths = []
        
    cdef void push(self,double cost, DirectedPath path):
        """Push a candidate path onto heap

        :param cost: the 
        :rtype: None
        """
        cdef list paths = self.sortedpaths
        cdef set seen = self.seen
        cdef tuple tpath = tuple(<np.ndarray>path.seq)

        if tpath not in seen:         
            heappush(paths,(cost,path))
            seen.add(tpath)

    def __iter__(self):
        return iter(self.sortedpaths)
        
    cdef DirectedPath pop(self):
        """Return the top item with lowest score

        :returns: a path instance 
        """
        cdef list paths = self.sortedpaths
        cdef double cost
        cdef DirectedPath popped

        cost,popped = heappop(paths)

        return <DirectedPath>popped

    cdef bint empty(self):
        """Returns if buffer is empty 

        :rtype: bool
        """
        cdef list paths = self.sortedpaths
        
        return <bint>(not paths)
    
    def __len__(self):
        ## number of candidates 
        cdef list paths = self.sortedpaths
        
        return <int>len(paths)

    def __bool__(self):
        cdef list paths = self.sortedpaths
        
        return <bint>(paths == [])


cdef class KShortestPaths:

    """Cython level class for storing k shortest path lists """

    def __init__(self,A,k):
        """Create a kshortestpaths instance 
        
        :param A: the current list (with blank items if k is larger)
        :param k: the the k found 
        """
        self._k = k
        self._A = A[:k] 

    property size:
        """The number of items found"""
        
        def __get__(self):
            """Returns the number of items found"""
            cdef int k = self._k
            return k
        
    property paths:
        """The list of current paths"""

        def __get__(self):
            """Return the current paths 

            :rtype: np.ndarray
            """
            cdef np.ndarray paths = self._A
            return paths

### c methods
        
@boundscheck(False)
cdef bint check_equal(np.ndarray[ndim=1,dtype=np.int32_t] array_1,
                         np.ndarray[ndim=1,dtype=np.int32_t] array_2,int size):
    """Checks if two arrays are equal

    :param array_1: the first array 
    :param array_2: the second array
    :param size: the size of the array 
    """
    if array_1[0] != array_2[0]: return False
    if array_1[-1] != array_2[-1]: return False
    for i in range(size):
        if array_1[i] != array_2[i]:
            return False
    return True

@boundscheck(False)
@cdivision(True)
cdef KShortestPaths dag_shortest_paths(DirectedAdj adj,double[:] weights,int k,int size):
    """Find the k-best paths for a dag using Yen's algorithm
    :param adj: the adjacency information for the graph 
    :param weights: the edge weights 
    :param k: the number of paths to returns
    :param size: the number of nodes 
    """
    cdef int[:] edges = adj.edges 
    cdef int[:,:] spans = adj.node_spans
    cdef np.ndarray A = np.ndarray((k,),dtype=np.object)
    cdef DirectedPath top_candidate,previous,spur_path,new_path,recent_path
    cdef PathBuffer B = PathBuffer()

    ## the different sequences involved
    cdef np.ndarray[ndim=1,dtype=np.int32_t] prev_seq, spur_seq,total_path,root_path,other_path
    cdef np.ndarray[ndim=1,dtype=np.double_t] prev_score,sput_score,total_node_score

    cdef double root_score,total_score
    cdef int i,j,current_k,prev_size,root_len
    cdef set ignore_nodes,ignore_edges 
    cdef bint equal
    cdef int root_outgoing,block_len
    cdef set root_block,observed_spurs
    cdef dict spurs_seen = <dict>defaultdict(set)
    
    ## equality lookup
    cdef int[:] equal_path = np.zeros((k,),dtype=np.int32)
    
    ## first shortest path 
    previous = dag_shortest(0,0.0,edges,spans,weights,size)
    A[0] = previous
    current_k = 1

    while True:

        ## did you find k already? 
        if current_k >= k: break

        ## reset equal path
        equal_path[:] = 0
            
        ## initialize what to ignore
        ignore_nodes = set()
        ignore_edges = set()

        ## previous size and actual sequence
        prev_size  = previous.size
        prev_seq   = previous.seq
        prev_score = previous.node_scores

        # ## go most recent best sequence

        for i in range(1,prev_size):
            root_path = prev_seq[:i]
            root_len = root_path.shape[0]

            ## find root score (to add to spur sequence)
            root_score = prev_score[root_len-1]

            ## number of outgoing
            root_outgoing = (spans[root_path[-1]][-1]+1)-spans[root_path[-1]][0]
            root_block = set()

            ## check out best paths
            for j in range(0,current_k):
                                
                ## recent path from k-best list so far 
                recent_path = A[j]
                other_path = recent_path.seq

                ## easy case, previous best path is the most recent 
                if j == (current_k - 1):
                    ignore_edges.add((other_path[i-1],other_path[i]))
                    root_block.add(other_path[i])

                ## starting point 
                elif i == 1 and root_path[0] == other_path[0]:
                    ignore_edges.add((other_path[i-1],other_path[i]))
                    root_block.add(other_path[i])
                    equal_path[j] = 1

                ## previously equal, new part equal?
                elif equal_path[j] == 1 and root_path[-1] == other_path[i-1]:
                    ignore_edges.add((other_path[i-1],other_path[i]))
                    root_block.add(other_path[i])

                ## not equal anymore 
                elif equal_path[j] == 1 and i > 1:
                    equal_path[j] = 0

            ignore_nodes.add(root_path[-1])
            block_len = len(root_block)
            
            ## check if all edges have been exhausted    
            if block_len == root_outgoing: continue

            ## check if restriction have already been looked at (Lawler's rule)
            #observed_spurs = spurs_seen[root_path[-1]]
            observed_spurs = spurs_seen[tuple(root_path)]
            if block_len <= len(observed_spurs): continue
            observed_spurs.update(root_block)

            ## do the next best path search
            spur_path = dag_shortest(root_path[-1],root_score,
                                         edges,spans,weights,size,
                                         ignored_edges=ignore_edges,
                                         ignored_nodes=ignore_nodes)

            ## is it a valid path, or did a block no allow for a path to end?
            if spur_path.invalid: continue

            ## sput path and score 
            spur_seq   = spur_path.seq
            spur_score = spur_path.node_scores

            ## glue together the root path and spur path
            total_path = np.concatenate((root_path,spur_seq[1:]),axis=0)
            total_score = spur_path.score
            total_node_score = np.concatenate((prev_score[:i],spur_score[1:]),axis=0)

            ## create new path and add to list 
            new_path = DirectedPath(total_path,total_node_score,
                                        total_score,
                                        total_path.shape[0])

            B.push(total_score,new_path)

        ## see if there are more candidates
        if B.empty(): break
            
        ## get the top candidate 
        top_candidate = B.pop()
        A[current_k] = top_candidate
        current_k += 1
        previous = top_candidate

    return KShortestPaths(A,current_k)
        
@boundscheck(False)
@wraparound(False)
cdef inline DirectedPath dag_shortest(int source,double start_score,
                                 int[:] edges,int[:,:] spans, ## graph components
                                 double[:] weights, ## edge weights 
                                 int size, # the number of nodes
                                 set ignored_edges=set(), ## edges not to consider
                                 set ignored_nodes=set(),
      ):
    """Finding a single shortest 
    :param source: the source node to start at 
    :param edges: the graph edges 
    :param spans: the number of graph spans 
    :param weights: the edge weights 
    :param size: the number of nodes 
    :param blocks: the edges to ignore (spur nodes)
    """
    cdef int i,j,start,end,node
    cdef double dd
    cdef double *d           = <double *>malloc(size*sizeof(double))
    cdef int *p              = <int *>malloc(size*sizeof(int))
    cdef double *node_scores = <double *>malloc(size*sizeof(double))
    cdef int *out_seq        = <int *>malloc(size*sizeof(int))
    cdef double weight
    cdef int current,seq_len
    cdef double cdef_float_infp = float('+inf')
    
    ## change to cpython arrays 

    try:
        with nogil:

            ## relax all nodes/initialize
            for i in range(size):
                d[i] = cdef_float_infp
                p[i] = -1
                out_seq[i] = -1
                node_scores[i] = cdef_float_infp

            d[source] = start_score

            for i in range(source,size-1):
                start = spans[i][0]
                end = spans[i][1]
            
                for j in range(start,end+1):
                    node = edges[j]

                    with gil:
                        ### eliminate ignored edges and edges
                        if node in ignored_nodes: continue
                        if (i,node) in ignored_edges: continue

                    weight = weights[j]
                
                    ### find smallest link 
                    if (d[node] > d[i] + weight):
                        d[node] = d[i] + weight
                        p[node] = i

            out_seq[0] = size-1
            node_scores[0] = d[size-1]
            current = size-1
            seq_len = 1
            weight = d[current]

            ## need to keep track of scores on each node
            while True:
            
                current = p[current]
                out_seq[seq_len] = current
                node_scores[seq_len] = d[current]
                if current <= source: break
                seq_len += 1

        ## check if it is invalid
        
        return DirectedPath(
            np.array([i for i in out_seq[:seq_len+1]],dtype=np.int32)[::-1],
            np.array([dd for dd in node_scores[:seq_len+1]],dtype='d')[::-1],
            weight,seq_len+1)

    finally:
        free(d); free(p)
        free(node_scores); free(out_seq)
        
def main(argv):
    logging.basicConfig(level=logging.INFO)

    EX = """
0	1	5
0	2	3
1	2	2
1	3	6
2	3	7
2	4	4
2	5	2
3	4	-1
3	5	1
4	5	-2
"""


    g = WeightedDAG.from_str(EX)
    paths = g.shortest_paths(0,k=12,trace=True)
    plen = paths.shape[0]

    for i in range(plen):
        print paths[i].path
    
if __name__ == "__main__":
    main(sys.argv[1:]) 
