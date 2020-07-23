#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:25:28 2020

@author: user
"""


import pickle
a=0
with open('a.pickle', 'wb') as f:
    pickle.dump([a], f)
a=[1,2,3]
with open('a.pickle', 'rb') as f:
    a= pickle.load(f)
a=[1,2,3]
with open('a.pickle', 'wb') as f:
    pickle.dump([a], f)
    
with open('a.pickle', 'rb') as f:
    a= pickle.load(f)