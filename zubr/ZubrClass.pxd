# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson

"""

cdef class ZubrClass:
    pass

cdef class ZubrLoggable(ZubrClass):
    pass

cdef class ZubrSerializable(ZubrLoggable):
    pass

cdef class ZubrConfigurable(ZubrSerializable):
    pass 
