import numpy as np
cimport numpy as np
from zubr.GraphDecoder cimport GraphDecoderBase
from zubr.Graph cimport WordGraph,DirectedAdj,Path
from zubr.ExecutableModel cimport ExecutableModel

from zubr.GraphDecoder cimport (
    SequencePath
)

from zubr.neural.Seq2Seq cimport (
    Seq2SeqModel,
    Seq2SeqLearner
)

cdef class NeuralSPDecoderBase(GraphDecoderBase):
    cdef public Seq2SeqLearner learner
    cdef WordGraph graph
    cdef np.ndarray edge_labels
    cdef dict edge_map
    cdef object _config

## decoder types 
    
cdef class NeuralSPDecoder(NeuralSPDecoderBase):
    pass 

cdef class ExecutableNeuralDecoder(NeuralSPDecoderBase):
    cdef ExecutableModel executor

## polyglot models

cdef class PolyglotExecutableNeuralDecoder(ExecutableNeuralDecoder):
    pass 

cdef class PolyglotSPNeuralDecoder(NeuralSPDecoderBase):
    cdef dict langs

## concurrent models

cdef class ConcurrentNeuralSPDecoder(NeuralSPDecoder):
    cpdef _setup_jobs(self,config)

cdef class NeuralPolyglotConcurrentDecoder(ConcurrentNeuralSPDecoder):
    cdef dict langs

cdef class NeuralConcurrentDecoder(ConcurrentNeuralSPDecoder):
    pass

## helpers

cdef class NeuralSequencePath(SequencePath):
    cdef np.ndarray state_seq
    cdef np.ndarray eos_encoding(self,int eos)
