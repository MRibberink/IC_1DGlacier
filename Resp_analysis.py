#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ice=pd.read_csv("ice_dimensions.csv")

def min_max_scaling(data):
    data_2= (data - np.min(data)) / (np.max(data) - np.min(data))
        
    return data_2
def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number-valor))

    return aux.index(min(aux))
    
    def resp_time(num,diff):
    eq_levels=[]
    loc_eq_levels=[]
    rt=np.arange(0,num)
    times=np.arange(0,30000)
    sections=np.zeros((num,len(times)))
    ice_h=list(ice['0'][50000:])
    e_fold=1/np.exp(1)
    changes=np.arange(diff*200,diff*num*200,diff*200)
    steps=np.arange(diff*100,diff*num*200,100000)
    
    for i in range(num):
        eq_levels=np.append(eq_levels,ice['0'][steps[i]])
        loc_eq_levels=np.append(loc_eq_levels,ice_h.index(eq_levels[i])+50000)
    for j in range(num-1):
        section=[]
        section=ice['0'][int(changes[j]):int(loc_eq_levels[j+1])]
        section=np.append(section,np.zeros((len(times)-np.shape(section)[0])))
        section_n=min_max_scaling(section)
        sections[j,:]=section_n
        
        rt[j]=times[closest(section_n,e_fold)]
        
    
    return sections,rt/200.