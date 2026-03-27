#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:28:23 2026

@author: antonioraphael
"""

import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("/Users/antonioraphael/Documents/PROJECT-CLONES/Data-Storage/AirQualityData/Final-Data/AirQualityData_Imputed_Final.csv")

boxcoxcols = list(data.columns[16:21]) + list(data.columns[31:105])

def boxcox_safe(x):
    x = x.astype(float)

    transformed, _ = stats.boxcox(x)

    return transformed
    
data[boxcoxcols] = data[boxcoxcols].apply(boxcox_safe)

data = data.drop(columns = ['Test'])


