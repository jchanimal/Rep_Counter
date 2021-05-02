# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
import math
from scipy.signal import find_peaks

def _compute_mean_features(window):
    return np.mean(window, axis=0)

# TODO: define functions to compute more features
def _compute_variance(window):
    return np.var(window, axis=0)

def _compute_rfft(window):
    ret = np.fft.rfft(window, axis=0)[0]
    return ret.astype(float)

def _compute_std(window):
    return np.std(window, axis=0)

def _compute_entropy(window):
    counts, bins = np.histogram(window, density=True)
    counts+=1
    
    PA = counts/ np.sum(counts, dtype=float)
    SA = -PA * np.log(PA)
    S = -np.sum(PA * np.log(PA), axis=0)
    
    return np.array(S, ndmin = 1)
   
        
def _compute_peak_length(window):
    ret = []
    for i in range(len(window[0])):
        x = float(window[0][i])
        y = float(window[1][i])
        z = float(window[2][i])
        r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        ret.append(r)
    mag = np.array(ret)
    peaks, _ = find_peaks(mag, prominence=1)
    return [len(peaks)]

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    
    x = []
    feature_names = []
    
    x.append(_compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    x.append(_compute_variance(window))
    feature_names.append("x_var")
    feature_names.append("y_var")
    feature_names.append("z_var")

    x.append(_compute_rfft(window))
    feature_names.append("x_rfft")
    feature_names.append("y_rfft")
    feature_names.append("z_rfft")
    
    x.append(_compute_peak_length(window))
    feature_names.append("numberOf_peaks")
    
    x.append(_compute_entropy(window))
    feature_names.append("entropy")
    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector