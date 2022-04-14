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


'''
Dynamic functions, the simplest approach:
 background noise just contributes to the potential.
  We assume it to contribute equally to all neurons,
   we do not take here into account variations in the noise, 
   we assume all the neurons to be identical.
   The growth of the EPSP is instanateous in this model.

The function that updates neuron's state based on its current state and input.

We are not modeling laser activated neurons as separate entities 
but just as the same neuronsvwith additional input

'''
@jit(nopython=True)
def neuron_state_update(state, inp, voltage, refr, threshold, EPSP_decay, tau_m):
    V = voltage * EPSP_decay + inp / tau_m
    if ((state < refr) or (V < threshold)):
        return(state + 1, V)
    else: 
        return(0,0)
@jit(nopython=True)
def neuron_state_update_noise(state, inp, voltage, refr, threshold, EPSP_decay, tau_m, noise_freq):
    V = voltage * EPSP_decay + inp / tau_m
    to_fire = np.random.rand()
    if (((state < refr) or (V < threshold)) and (to_fire > noise_freq) ):
        return(state + 1, V)
    else: 
        return(0,0)
    
@jit(nopython=True)    
def neuron_output(state,n_neighbors, W_synapse, tau_s, mu,  sigma):
    for i in range(n_neighbors):
        if (state == 0):
            W_synapse[i] = np.random.lognormal(mu, sigma)
            #print(W_synapse[i])
    return(W_synapse*state*np.exp(-state / tau_s), W_synapse)
    
    

#It is not real excitation, but effective one
@jit(nopython=True) 
def excitation_by_laser(V, mu, sigma, n_spikes, n_timesteps):
    output = np.zeros(n_timesteps)
    j = 0
    for i in range(n_spikes):
        per = int(np.random.lognormal(mu, sigma))
        j = j + per
        output[j] = V
    return(output)




@jit(nopython = True)
def one_step_LIF(state, input_matrix, voltage,syn_del, W_syn,  edges, t, parameters):
    EPSP_decay = parameters['EPSP_decay'] # we multiply voltage to this constant every step to get exponential decay
    refr = int(parameters['refr']) 
    mu = parameters['mu'] 
    sigma = parameters['sigma']
    tau_s = int(parameters['tau_s'])
    tau_m = int(parameters['tau_m'])
    threshold = parameters['threshold'] #threshold for the transition
    n_neurons = int(parameters['n_neurons']) #total number of neurons
    
    for i in range(n_neurons):
        statei = state[i,t]
        n_neighbors = int(edges[i,0])
        W_synapse = np.zeros(n_neighbors)
        out = np.zeros(n_neighbors)
        constant = statei*np.exp(-statei / tau_s)
        for j in range(n_neighbors):
            out[j] = W_syn[i,j]*constant
        #out, W_synapse = neuron_output(statei,n_neighbors, W_synapse, tau_s, mu,  sigma)
            neighbor = int(edges[i, j+1])
            delay = int(syn_del[i,j])
            input_matrix[neighbor, t + delay + 1] = input_matrix[neighbor, t + delay + 1] + out[j]
    for i in range(n_neurons):
        state[i, t + 1], voltage[i, t + 1] = neuron_state_update(state[i, t ], input_matrix[i, t], voltage[i, t ], refr, threshold, EPSP_decay,tau_m)
        
    return(state, input_matrix, voltage)

@jit(nopython = True)
def create_noise_vector(N, fraction):
  nv = np.zeros(N)
  for i in range(N):
    p = np.random.rand()
    if (p < fraction):
      nv[i] = 1
  return(nv)

@jit(nopython = True)
def one_step_LIF_noise(state, input_matrix, voltage,syn_del, W_syn,  edges, t, parameters, noise_freq, noise_vector):
    
    EPSP_decay = parameters['EPSP_decay'] # we multiply voltage to this constant every step to get exponential decay
    refr = int(parameters['refr']) 
    mu = parameters['mu'] 
    sigma = parameters['sigma']
    tau_s = int(parameters['tau_s'])
    tau_m = int(parameters['tau_m'])
    threshold = parameters['V_threshold'] #threshold for the transition
    n_neurons = int(parameters['n_neurons']) #total number of neurons
  
    for i in range(n_neurons):
    
        statei = state[i,t]
        n_neighbors = int(edges[i,0])
        W_synapse = np.zeros(n_neighbors)
        out = np.zeros(n_neighbors)
        constant = statei*np.exp(-statei / tau_s)
        for j in range(n_neighbors):
            out[j] = W_syn[i,j]*constant
        #out, W_synapse = neuron_output(statei,n_neighbors, W_synapse, tau_s, mu,  sigma)
            neighbor = int(edges[i, j+1])
            delay = int(syn_del[i,j])
            input_matrix[neighbor, t + delay + 1] = input_matrix[neighbor, t + delay + 1] + out[j]
    for i in range(n_neurons):
      if (noise_vector[i] == 1):
        state[i, t + 1], voltage[i, t + 1] = neuron_state_update_noise(state[i, t ], input_matrix[i, t], voltage[i, t ], refr, threshold, EPSP_decay, tau_m, noise_freq)
      else:
        state[i, t + 1], voltage[i, t + 1] = neuron_state_update(state[i, t ], input_matrix[i, t], voltage[i, t ], refr, threshold, EPSP_decay, tau_m)
  

    return(state, input_matrix, voltage)

@jit(nopython=True)
def prepare_syndel(average, std, stepsize, edges, n_neurons ):
    mu, sigma = prepare_lognormal(average, std, stepsize)
    synaptic_del = np.zeros(n_neurons**2).reshape(n_neurons, n_neurons)
    for i in range(n_neurons):
        for j in range(n_neurons - 1):
            if (edges[i, j + 1] > 0):
                 syndel = np.random.lognormal(mu, sigma)
    return(synaptic_del)





@jit(nopython = True)
def prepare_input(V, period, std, stepsize, n_neurons, n_timesteps, n_spikes, set_activated):
    input_matrix = np.zeros(n_neurons*n_timesteps).reshape(n_neurons, n_timesteps)
    n_activated = len(set_activated)
    mu, sigma = prepare_lognormal(period, std, stepsize)
    for i in range(n_activated):
        vec = excitation_by_laser(V, mu, sigma, n_spikes, n_timesteps)
        number = int(set_activated[i])
        for j in range(n_timesteps):
            input_matrix[number, j] = vec[j]
    return(input_matrix)
        
    

        
        
@jit(nopython=True)
def full_process(edges, syn_del, input_matrix, W_matrix,  par,  noise_vector):
    n_neurons = int(par['n_neurons'])
    noise_freq = par['Noise Frequency']
    stepsize = par['stepsize']
    n_timesteps = int(par['n_timesteps'])
    tau_s = int(par['tau_s'])
    states = np.zeros(n_neurons*n_timesteps).reshape(n_neurons, n_timesteps)
    for i in range(n_neurons):
        for j in range(n_timesteps):
            states[i,j] = 1000 * tau_s
    max_del = np.max(syn_del.reshape(n_neurons**2))
    V_threshold = par['V_threshold'] 
    voltages = np.zeros(n_neurons*n_timesteps).reshape(n_neurons, n_timesteps)
    t = 0
    av_V = 0
    while ((t < n_timesteps - 1 - max_del) and (av_V < V_threshold)):
       # print(3)
        states, input_matrix, voltages = one_step_LIF_noise(states, 
        input_matrix, voltages, syn_del, W_matrix,  edges, t, par, 
        noise_freq, noise_vector)
      #  print(4)
        t = t + 1
        sum_V = 0
        for i in range(n_neurons):
            sum_V = sum_V + voltages[i,t]
        av_V = sum_V / n_neurons    
        #print(av_V)
    return(states, voltages, t) 
                







    

    
    



def whospikes(N, n_timesteps, states):
    for i in range(n_timesteps):
        for j in range(N):
            if (states[j,i] == 0):
                print(i, "spikes", j)


        
@jit(nopython=True)           
def prepare_lognormal(av, std,stepsize):   
    sigmasq = np.log((std/av)**2 + 1) 
    sigma = np.sqrt(sigmasq)
    mu = np.log(av / stepsize)  - sigmasq / 2.0
    return(mu,sigma)    
    

    

    
@jit(nopython = True)
def prepare_input2(params, set_activated):
    n_neurons = int(params['n_neurons'])
    n_timesteps = int(params['n_timesteps'])
    #print(1)
    input_matrix = np.zeros(n_neurons*n_timesteps).reshape(n_neurons, n_timesteps)
    n_activated = len(set_activated)
    for i in range(n_activated):
        vec = excitation_by_laser_2(params)
        number = int(set_activated[i])
        for j in range(n_timesteps):
            input_matrix[number, j] = vec[j]
    return(input_matrix)  

@jit(nopython=True) 
def excitation_by_laser_2(params):
    
    n_timesteps = int(params['n_timesteps'])
    output = np.zeros(n_timesteps)
    mdellas = params['Mean delay for the first spike from laser activation']
    stdellas = params['Standard deviation for the delay of the first spike']
    V = params['Laser strength']
    period = params['Laser mean period']
    deviation = params['Standard deviation of laser period']
    n_spikes = params['Number of spikes']
    j = int(np.random.normal(mdellas, stdellas))
    output[j] = V
    for i in range(n_spikes - 1):
        per = int(np.random.normal(period, deviation))
        j = j + per
        output[j] = V
    return(output)




