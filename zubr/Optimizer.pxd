from zubr.ZubrClass cimport ZubrSerializable
from zubr.Dataset cimport Data,RankDataset,RankScorer

## model types

cdef class OptimizerBase(ZubrSerializable):
    cdef public object config

## new classes

cdef class RankOptimizer(OptimizerBase):
    cdef object extractor
    cdef object model,_best_model
    cdef int _improvement
    cdef int _train_model(self,RankDataset dataset,RankDataset valid) except -1
    cdef RankScorer _test_model(self,RankDataset dataset,str ttype,int it=?,bint debug=?)
    cdef RankScorer vscore

## online learner

cdef class OnlineOptimizer(RankOptimizer):
    cdef inline void log_iter(self,i,start_time,ll)
    cdef inline void log_test(self,num,ttype,start_time,i)
    #cdef RankScorer vscore
    
# batch learner

cdef class BatchOptimizer(RankOptimizer):
    pass

## minimatch learner 

cdef class MiniBatchOptimizer(RankOptimizer):
    pass 
