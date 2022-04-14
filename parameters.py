
import numpy as np


from numba import jit #acceleration of the code
from numba import njit
from numba.core import types
from numba.typed import Dict

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
import copy
import networkx as nx
from copy import copy, deepcopy

import pandas as pd

@njit
def my_params( ):
    stepsize = 0.05 
    #the size of the time step, in milliseconds
    
    refr = 3 / stepsize 
    # refractory period of the neuron. Number before division is in milliseconds 
    
    tau_s = 0.5 / stepsize 
    #tau_s from the equation for the current I = W * t * exp(-t/ tau_s). Number in ms.
    
    tau_m = 25 / stepsize 
    #tau_m membrane time constant in milliseconds
    
    V_threshold = 12.3 
    #the threshold voltage in millivolts
    EPSP_decay = np.exp(-1.0 / tau_m )
    # the factor by which the voltage decays following the equations
    
    V_EPSP = 2.8 
    #average magnitude of EPSP in mV
    
    std_EPSP = 1.5 
    #standard deviation of EPSP in mV
    
    W_average = 300 
    #approximate value of the parameter W in the equation for current (see above) in mV
    
    sigmasq = np.log((std_EPSP/V_EPSP)**2 + 1) 
    #square of parameter sigma in lognormal distribution 
    
    sigma = np.sqrt(sigmasq) 
    #parameter sigma in lognormal distribution
    
    mu = np.log(W_average) - sigmasq / 2.0 
    #parameter mu in lognormal distribution
    
    mu = mu + np.log(stepsize) 
    #changing the stepsize goes into mu in this way
    
    noise_freq = 0.0005 * stepsize 
    # frequency over the noise in kHz (1/ms)
    
    avdel = 1.3 / stepsize  
    #average synaptic delay time. In ms before division
    
    stddel = 1.1 / stepsize 
    #standard deviation of the synaptic delay time. In ms before division
    
    n_neurons = 1000
    #number of neurons
    
    n_timesteps = 10000 
    #the number of steps in time

    parameters = dict()
    parameters['EPSP_decay'] = EPSP_decay
    parameters['refr'] = refr
    parameters['mu'] = mu
    parameters['n_timesteps'] = n_timesteps
    parameters['sigma'] = sigma 
    parameters['tau_s'] = tau_s
    parameters['tau_m'] = tau_m
    parameters['V_threshold'] = V_threshold
    parameters['n_neurons'] = n_neurons
    parameters['stepsize'] = stepsize
    parameters['ER connection probability'] = 0.065
    parameters['Average synaptic delay'] = 1.3 / stepsize
    parameters['Standard deviation for synaptic delay'] = 1.1 / stepsize
    parameters['Noise Frequency'] = noise_freq
    parameters['Laser strength'] = 20 * tau_m 
    #magnitude of the input. Should be enough to make neuron to fire 
    
    parameters['Laser mean period'] = 39.2 / stepsize 
    #average period of spiking after laser stimulation in ms, 
    #taken from frequency in Kaiwen paper
    
    parameters['Standard deviation of laser period'] = 4.7 / stepsize 
    #standard deviation  of the period
    
    parameters['Number of spikes'] = 7 
    #number of spikes each stimulated neuron produces before stopping to spike, 
    #based on Kaiwen's paper
    
    parameters['Number of laser activated neurons'] = 7
    #number of activated neurons
    
    parameters['Mean delay for the first spike from laser activation'] = 20 / stepsize
    
    parameters['Standard deviation for the delay of the first spike'] = 3 / stepsize
    
    return parameters