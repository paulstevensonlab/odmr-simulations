#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:21:19 2023

@author: nicolevoce
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import yaml
import os
from scipy import signal
from scipy.stats import pearsonr
from scipy.fft import fft, ifft, fftfreq
from lmfit.models import Model
from lmfit import Parameters
import sympy as sp

def simulate_ODMR(x, FWHM1, w1, FWHM2, w2, c, a):
    L = a - a * c * ((FWHM1 / 2) ** 2 / ((x - w1) ** 2 + (FWHM1 / 2) ** 2) + (FWHM2 / 2) ** 2 / ((x - w2) ** 2 + (FWHM2 / 2) ** 2))
    L_noise = np.random.poisson(L)
    return L, L_noise

def derivative(x_array, FWHM1, w1, FWHM2, w2, c, a):
    x = sp.symbols('x')
    L = a - a * c * ((FWHM1 / 2) ** 2 / ((x - w1) ** 2 + (FWHM1 / 2) ** 2) + (FWHM2 / 2) ** 2 / ((x - w2) ** 2 + (FWHM2 / 2) ** 2))
    dL_dx = sp.diff(L, x)
    dL_dx_array = np.array([dL_dx.subs(x, val) for val in x_array], dtype=float)
    max_slope_index = np.argmax(dL_dx_array)
    point_of_max_slope = x_array[max_slope_index]
    return max_slope_index, point_of_max_slope
        

def magnetic_field(amp, freq, t, c = 20):
    mag = amp * np.sin(2 * np.pi * freq * t) + c
    return mag

def noisy_magnetic_field(white_noise, scaling):
    import numpy as np
    white_noise_fft = fft(white_noise) * fft(scaling)
    noisy_mag_field = ifft(white_noise_fft)
    return np.real(noisy_mag_field)


def frequencies(B):
    w1 = 2.87 + 2.8e-3 * B
    w2 = 2.87 - 2.8e-3 * B
    return w1, w2

def autocorrelation(test, time):
    import numpy as np
    autocorr = np.correlate(test, test,mode='full')[(time.shape[0]-1):]
    return autocorr

def norm_autocorrelation(autocorr):
    import numpy as np
    return np.mean(autocorr)/(np.mean(autocorr)[0])

def fit(x, a, T):
    import numpy as np
    return a*np.exp(-x/T)

def arrays(amp, tau, time, contrast, counts, FWHM, freq):
    import numpy as np
    mag_field_arrays = np.zeros((amp.size, tau.size, time.size))
    noisy_mag_field_arrays = np.zeros((amp.size, tau.size, time.size))

    odmr_spectrum_array = np.zeros((amp.size, tau.size, time.size, 
                                contrast.size, counts.size, FWHM.size, freq.size)) #time.size=mag_field.size

    noisy_odmr_spectrum_array = np.zeros((amp.size, tau.size, time.size, 
                          contrast.size, counts.size, FWHM.size, freq.size))

    noise_odmr_spectrum_array = np.zeros((amp.size, tau.size, time.size, 
                                contrast.size, counts.size, FWHM.size, freq.size))

    noise_noisy_odmr_spectrum_array = np.zeros((amp.size, tau.size, time.size, 
                          contrast.size, counts.size, FWHM.size, freq.size))

    rms_values = np.zeros((amp.size, counts.size))
    difference_values = np.zeros((amp.size, counts.size, time.size))
    
    return mag_field_arrays, noisy_mag_field_arrays, odmr_spectrum_array, noisy_odmr_spectrum_array, noise_odmr_spectrum_array, noise_noisy_odmr_spectrum_array, rms_values, difference_values

