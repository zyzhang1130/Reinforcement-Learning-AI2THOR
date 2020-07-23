#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:58:19 2020

@author: user
"""
this = 1
that = 2
from globals import *
import os
#os.rename( "globals.py", "globals.bak" )
with open( "globals.py", "w" ) as target:
    for variable in ('some', 'list', 'of', 'sensible', 'globals'):
        target.write( "{0!s} = {1!r}".format( variable, globals()[variable] ))