from zubr.ZubrClass cimport ZubrSerializable

cdef class ExecutableModel(ZubrSerializable):
    cdef public object proc

cdef class PromptingSubprocess(ExecutableModel):
    cdef str start_up
    cdef double timeout

cdef class LispModel(ExecutableModel):
    pass

## example models

cdef class GeoExecutor(PromptingSubprocess):
    pass 
