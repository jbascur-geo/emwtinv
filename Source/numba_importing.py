# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:12:38 2024

@author: juan
"""
import os
# note that this must be executed before 'import numba'
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
import numba
