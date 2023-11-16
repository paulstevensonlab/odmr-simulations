#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
import flos_backend as fcs
import argparse


# In[2]:


hf = h5py.File('data.h5', 'w')
class FLOS:
    def __init__(self, time, x_range, itr=1000, tau=50, amp=0.5, contrast_values=0.2, counts_values=10, FWHM=0.002):
        self.itr = itr
        self.time = time
        self.x_range = x_range
        self.tau = tau
        self.amp = amp
        self.contrast_values = contrast_values
        self.counts_values = counts_values
        self.FWHM = FWHM


    def run_sim(self, save_on = True):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=False, sharey=False) 
        if save_on:

            hf.create_dataset('parent_contrast_values', data=self.contrast_values)
            hf.create_dataset('parent_counts_values', data=self.counts_values)
            hf.create_dataset('parent_FWHM_values', data=self.FWHM)
            hf.create_dataset('parent_tau_values', data=self.tau)
            hf.create_dataset('parent_amplitude_values', data=self.amp)

        acf_norm_mag_array, acf_norm_odmr_array1, acf_norm_odmr_array2, autocorr_mag_data, autocorr_odmr_data1, autocorr_odmr_data2, noisy_odmr_autocorr_1, noisy_odmr_autocorr_2, photon_trajectory, noisy_w_1_array, noisy_w_2_array = fcs.arrays(self.itr, self.time)

        freq_1, freq_2 = fcs.freq_point(self.x_range, self.FWHM)

        for w in range(self.itr):
            noisy_w_1, noisy_w_2, noisy_mag_fields, noisy_w_1_array, noisy_w_2_array = fcs.freq_func(self.itr, self.tau, self.time, self.amp, noisy_w_1_array, noisy_w_2_array, w)

            noise_odmr_spectrum, noise_noisy_odmr_spectrum = fcs.simulate_ODMR(freq_1, 
                                                                            self.FWHM, np.asarray(noisy_w_1_array[w, :]), self.FWHM, np.asarray(noisy_w_2_array[w, :]), self.contrast_values, self.counts_values) # you can pass the array directly to the function to avoid another nested loop
                                                                                                                            
                                                                                                                                                # and we only need to calculate the count rate at one frequency
            photon_trajectory[w, :] = np.asarray(noise_noisy_odmr_spectrum)

            acf_norm_mag, acf_norm_odmr1, acf_norm_odmr2 = fcs.corr_analysis(noisy_mag_fields, self.time, photon_trajectory, w, 
                        autocorr_mag_data, autocorr_odmr_data1, autocorr_odmr_data2, 
                        acf_norm_mag_array, acf_norm_odmr_array1, acf_norm_odmr_array2)
            
            


            axes[0, 0].plot(self.time, acf_norm_mag, 'o')
            axes[0, 0].set_title('magnetic field correlation')

            axes[0, 1].plot(self.time, acf_norm_odmr1, 'o')
            axes[0, 1].set_title('w1 odmr correlation')
            
            axes[0, 2].plot(self.time, acf_norm_odmr2, 'o')
            axes[0, 2].set_title('w2 odmr correlation')

            axes[0, 0].set_xlim(0,2000)
            axes[0, 1].set_xlim(0,2000)
            axes[0, 2].set_xlim(0,2000)

        average_mag_array, average_odmr_array1, average_odmr_array2, avg_noisy_mag_autocorr_fit, avg_noisy_odmr_autocorr_fit1, avg_noisy_odmr_autocorr_fit2 = fcs.avg_corr_analysis(acf_norm_mag_array, acf_norm_odmr_array1, acf_norm_odmr_array2, self.tau, self.time)


        axes[1, 0].plot(self.time[1:], avg_noisy_mag_autocorr_fit.best_fit, '--', label = f"Fit, T_fit={avg_noisy_mag_autocorr_fit.params['e_decay'].value:.3f}, \n T_char={self.tau:.3f}")
        axes[1, 0].plot(self.time[1:], average_mag_array[1:], 'o', label = f'Corr, T_char={self.tau:.3f}')
        axes[1, 0].set_title('magnetic field correlation')
        axes[1, 0].legend()

        axes[1, 1].plot(self.time[1:], avg_noisy_odmr_autocorr_fit1.best_fit, '--', label = f"Fit, T_fit={avg_noisy_odmr_autocorr_fit1.params['e_decay'].value:.3f}, \n T_char={self.tau:.3f}")
        axes[1, 1].plot(self.time[1:], average_odmr_array1[1:], 'o', label = f'Corr, T_char={self.tau:.3f}')
        axes[1, 1].set_title('w1 odmr correlation')
        axes[1, 1].legend()

        axes[1, 2].plot(self.time[1:], avg_noisy_odmr_autocorr_fit2.best_fit, '--', label = f"Fit, T_fit={avg_noisy_odmr_autocorr_fit2.params['e_decay'].value:.3f}, \n T_char={self.tau:.3f}")
        axes[1, 2].plot(self.time[1:], average_odmr_array2[1:], 'o', label = f'Corr, T_char={self.tau:.3f}')
        axes[1, 2].set_title('w2 odmr correlation')
        axes[1, 2].legend()

        axes[1, 0].set_xlim(0,1000)
        axes[1, 1].set_xlim(0,1000)
        axes[1, 2].set_xlim(0,1000)
            
        axes[1, 3].plot(self.time, photon_trajectory[0,:], 'o', alpha = 0.2)
        axes[1, 3].set_title('photon trajectory')

       
        if save_on:
        
            hf.create_dataset('freq_1_pos', data=freq_1)
            hf.create_dataset('freq_2_pos', data=freq_2)
            hf.create_dataset('average_magfield_autocorrelation', data = average_mag_array)
            hf.create_dataset('average_magfield_autocorrelation_fit', data = avg_noisy_mag_autocorr_fit.best_fit)
            hf.create_dataset('average_odmr_autocorrelation_1', data = average_odmr_array1)
            hf.create_dataset('average_odmr_autocorrelation_2', data = average_odmr_array2)
            hf.create_dataset('average_odmr_autocorrelation_fit_1', data = avg_noisy_odmr_autocorr_fit1.best_fit)
            hf.create_dataset('average_odmr_autocorrelation_fit_2', data = avg_noisy_odmr_autocorr_fit2.best_fit)
            hf.create_dataset('avg_magfield_fit_characteristic_diffusion_time', data = avg_noisy_mag_autocorr_fit.params['e_decay'].value)
            hf.create_dataset('avg_odmr1_fit_characteristic_diffusion_time', data = avg_noisy_odmr_autocorr_fit1.params['e_decay'].value)
            hf.create_dataset('avg_odmr2_fit_characteristic_diffusion_time', data = avg_noisy_odmr_autocorr_fit2.params['e_decay'].value)
            hf.close()
                
            

def main():

    parser = argparse.ArgumentParser(description='Run FLOS simulation with specified parameters.')
    parser.add_argument('-i', type=int, default = 1000, required=True,  help = 'define # iteration')
    parser.add_argument('-ct', type=float, default = 0.2, required=True,  help = 'define contrast value')
    parser.add_argument('-c', type=float, default = 10, required=True, help = 'define raw counts value')
    parser.add_argument('-f', type=float, default = 0.002, required=True, help = 'define FWHM value')
    parser.add_argument('-t', type=float, default = 50, required=True, help = 'define characteristic diffusion time')
    parser.add_argument('-a', type=float, default = 0.5, required=True, help = 'define amplitude of noisy mag field')
    args = parser.parse_args()

    time = np.linspace(0.0, 5e04, 5001)
    x_range = np.linspace(2.7, 3.0, time.size)
    flos = FLOS(time, x_range, args.i, args.t, args.a, args.ct, args.c, args.f)


    flos.run_sim()
if __name__ == '__main__':
    main()

