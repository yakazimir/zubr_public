#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is subject to the terms and conditions defined in
file 'LICENSE', which is part of this source code package.

author : Kyle Richardson
"""

## main exception class for utlities

class ZubrUtilError(Exception):
    def __init__(self,msg):
        self.msg = msg

    def __str__(self):
        return self.msg    
