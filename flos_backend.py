#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:21:19 2023

@author: nicolevoce
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import lmfit as lm
# from lmfit import Parameters

def simulate_ODMR(x, FWHM1, w1, FWHM2, w2, c, a):
    L = a - a * c * ((FWHM1 / 2) ** 2 / ((x - w1) ** 2 + (FWHM1 / 2) ** 2) + (FWHM2 / 2) ** 2 / ((x - w2) ** 2 + (FWHM2 / 2) ** 2))
    L_noise = np.random.poisson(L)
    return L, L_noise
        

def magnetic_field(amp, freq, t, c = 20):
    mag = amp * np.sin(2 * np.pi * freq * t) + c
    return mag

def noisy_magnetic_field(white_noise, scaling):
    white_noise_fft = fft(white_noise) * fft(scaling)
    noisy_mag_field = ifft(white_noise_fft)
    return np.real(noisy_mag_field)


def frequencies(B):
    w1 = 2.87 + 2.8e-3 * B
    w2 = 2.87 - 2.8e-3 * B
    return w1, w2

def autocorrelation(test, time):
    autocorr = signal.correlate(test, test, mode = 'full', method = 'fft')[(time.shape[0]-1):]
    return autocorr

def fit(x, a, T):
    return a*np.exp(-x/T)

def freq_point(x_range, FWHM):
    max_slope_index_1 = np.argmin(np.abs(x_range - (2814 - 0.6*0.5*1000*FWHM)/1000))
    max_slope_index_2 = np.argmin(np.abs(x_range - (2814 - 0.6*0.5*1000*FWHM)/1000))
    freq_1 = x_range[max_slope_index_1]
    freq_2 = x_range[max_slope_index_2]
    return freq_1, freq_2

def arrays(itr, time):
    acf_norm_mag_array = []
    acf_norm_odmr_array1 = []
    acf_norm_odmr_array2 = []


    autocorr_mag_data = np.empty((itr, time.shape[0]))
    autocorr_odmr_data1 = np.empty((itr, time.shape[0]))
    autocorr_odmr_data2 = np.empty((itr, time.shape[0]))

    noisy_odmr_autocorr_1 = [[] for _ in range(itr)]
    noisy_odmr_autocorr_2 = [[] for _ in range(itr)]

    photon_trajectory = np.zeros((itr, time.size))

    noisy_w_1_array = np.empty((itr, time.size))
    noisy_w_2_array = np.empty((itr, time.size))
    return acf_norm_mag_array, acf_norm_odmr_array1, acf_norm_odmr_array2, autocorr_mag_data, autocorr_odmr_data1, autocorr_odmr_data2, noisy_odmr_autocorr_1, noisy_odmr_autocorr_2, photon_trajectory, noisy_w_1_array, noisy_w_2_array

def freq_func(itr, tau, time, amp, noisy_w_1_array, noisy_w_2_array, _):

    noisy_mag_field_itr = []
    noisy_w_1 = [[] for _ in range(itr)]
    noisy_w_2 = [[] for _ in range(itr)]

    scaling = amp*np.array([np.exp(-t/tau) for t in time])
    white_noise = np.random.normal(size=time.size)

    noisy_mag_field = 0.1*noisy_magnetic_field(white_noise, scaling) + 20
    noisy_mag_field_itr.append(noisy_mag_field)

    for n, B_noise in enumerate(noisy_mag_field):
        noisy_w1, noisy_w2 = frequencies(B_noise)

        noisy_w_1[_].append(noisy_w1)
        noisy_w_2[_].append(noisy_w2)
       

    noisy_w_1_array[_, :] = noisy_w_1[_]
    noisy_w_2_array[_, :] = noisy_w_2[_]


    return noisy_w_1, noisy_w_2, noisy_mag_field, noisy_w_1_array, noisy_w_2_array
    
def corr_analysis(noisy_mag_fields, time, photon_trajectory, _, 
                  autocorr_mag_data, autocorr_odmr_data1, autocorr_odmr_data2, 
                  acf_norm_mag_array, acf_norm_odmr_array1, acf_norm_odmr_array2):

    noisy_mag_autocorr = autocorrelation(noisy_mag_fields - np.mean(noisy_mag_fields), time)
    noisy_odmr_autocorr1 = autocorrelation(photon_trajectory[_]- np.mean(photon_trajectory[_]), time)
    noisy_odmr_autocorr2 = autocorrelation(photon_trajectory[_]- np.mean(photon_trajectory[_]), time)
   
    autocorr_mag_data[_, :]  = noisy_mag_autocorr
    autocorr_odmr_data1[_, :] = noisy_odmr_autocorr1
    autocorr_odmr_data2[_, :] = noisy_odmr_autocorr2



    acf_norm_mag = np.mean(autocorr_mag_data,0)/np.mean(autocorr_mag_data,0)[0]
    acf_norm_odmr1 = np.mean(autocorr_odmr_data1,0)/np.mean(autocorr_odmr_data1,0)[0]
    acf_norm_odmr2 = np.mean(autocorr_odmr_data2,0)/np.mean(autocorr_odmr_data2,0)[0]


    acf_norm_mag_array.append(acf_norm_mag)
    acf_norm_odmr_array1.append(acf_norm_odmr1)
    acf_norm_odmr_array2.append(acf_norm_odmr2)


    return acf_norm_mag, acf_norm_odmr1, acf_norm_odmr2

def avg_corr_analysis(acf_norm_mag_array, acf_norm_odmr_array1, acf_norm_odmr_array2, tau, time):
    stacked_mag_arrays = np.stack(acf_norm_mag_array, axis=0)
    average_mag_array = np.mean(stacked_mag_arrays, axis=0)

    stacked_odmr_arrays1 = np.stack(acf_norm_odmr_array1, axis=0)
    average_odmr_array1 = np.mean(stacked_odmr_arrays1, axis=0)

    stacked_odmr_arrays2 = np.stack(acf_norm_odmr_array2, axis=0)
    average_odmr_array2 = np.mean(stacked_odmr_arrays2, axis=0)

    model = lm.models.ExponentialModel(prefix = 'e_')
    pars = model.make_params()
    pars['e_amplitude'].set(value = 1.0, min = 0.0, vary = True)
    pars['e_decay'].set(value = tau, min = 0.0)
    avg_noisy_mag_autocorr_fit = model.fit(average_mag_array[1:], x = time[1:], params = pars)
    avg_noisy_odmr_autocorr_fit1 = model.fit(average_odmr_array1[1:], x = time[1:], params = pars)
    avg_noisy_odmr_autocorr_fit2 = model.fit(average_odmr_array2[1:], x = time[1:], params = pars)
    return average_mag_array, average_odmr_array1, average_odmr_array2, avg_noisy_mag_autocorr_fit, avg_noisy_odmr_autocorr_fit1, avg_noisy_odmr_autocorr_fit2