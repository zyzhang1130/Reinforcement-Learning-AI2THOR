#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:27:24 2020

@author: user
"""

a=0
b=1

import pickle
with open('train.pickle', 'wb') as f:
    pickle.dump([a,b], f)
    
    
with open('train.pickle', 'rb') as f:
    X_train, y_train = pickle.load(f)