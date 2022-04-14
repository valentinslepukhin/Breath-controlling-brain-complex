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



#Erdos-Renyi connectivity matrix 
@jit(nopython = True)
def getRandomConnectivity(N, pct_connected):
    #Directed mapping
    M = np.random.rand(N**2).reshape(N,N)
    for i in range(N):
        for j in range(N):
            if (i == j):
                M[i,j] = 0
            else:
                if (M[i,j] < pct_connected):
                    M[i,j] = 1 #i connects to j
                else:
                    M[i,j] = 0
    return M

@jit(nopython = True)
def kill_reciprocal(N,M):
  for i in range(N):
    for j in range(i):
      if ((M[i,j] == 1) and (M[j,i] == 1)):
        q =np.random.randint(2)
        if (q==0):
          M[i,j] = 0
        else:
          M[j,i] = 0
  return(M)






#Addition of motives by completing the triangle
@jit(nopython=True)
def addmotives(N,M,l,nummotives):
    for m in range(nummotives):
        c = 0
        while (c == 0):
            a = 0
            while(a < 2):
                i = np.random.randint(N)
                a = int(l[i,0])
            j1 = 0
            j2 = 0
            while (j1 == j2):
                j1 = np.random.randint(a)+1
                j2 = np.random.randint(a)+1
            b = int(l[j1,0])
            d = 0
            for k in range(b):
                if (j2 == int(l[j1,k+1])):
                    d = 1
            if (d==0):
                l[j1,0] = l[j1,0] + 1
                l[j1,b + 1] = j2
                c=1
    return(l)
        


#Addition of motives by adding full triangle    
def addmotives2(N,M,nummotives):
    m = 0
    while(m<nummotives):
        i = np.random.randint(N)
        j = np.random.randint(N)
        k = np.random.randint(N)
        if (M[i,j]==0):
            M[i,j] = 1
            m = m + 1
        if (M[j,k]==0):
            M[j,k] = M[j,k] + 1
            m = m + 1
        if (M[k,i] == 0):
            M[k,i] = M[k,i]+1
            m = m + 1
    #l = matrixOfEdges(M,N)
    return(M)        

def numberofmotives(N,M,l):
    num = 0
    for i in range(N):
        a = int(l[i,0])
        for j in range(a):
            for k in range(a):
                l1 = int(l[i,j+1])
                l2 = int(l[i,k+1])
                if (M[l1,l2]==1):
                    num = num + 1
    return(num)





def findandkillmotives(N,M,l,numtokill):
    num = 0
    i = 0
    while ((i<N)and(num < numtokill)):
        a = int(l[i,0])
        for j in range(a):
            for k in range(a):
                l1 = int(l[i,j+1])
                l2 = int(l[i,k+1])
                if (M[l1,l2]==1):
                    num = num + 1
                    M[l1,l2] = 0
        i = i + 1
    return(M,num)
    
def getrandomvector(N,ics):
    V=zeros(N)
    p=ics*1.0/N
    for i in range(N):
        if (np.random.rand(1)<p):
            V[i]=1
    return(V)



@jit(nopython=True)
def matrixOfEdges(M,N):
    N = int(N)
    E=np.zeros(N**2).reshape(N,N)  #prepare array with zeros
    k=np.sum(M,0) #array with degree of each vertex
    for i in range(N):
        a=int(k[i]) #degree of the current vertex
        E[i,0]=a  #we put it to the zero row of matrix of edges
    for i in range(N):
        a=int(k[i])
        q=1
        for j in range(N):
            if (M[j,i]==1):
                E[i,q]=j #all the next elements in current column are number of vertices current vertex is connected to
                q=q+1
    return(E)


@jit(nopython=True)
def recovermatrixfromlist(E,N):
    q=0;
    M=np.zeros(N*N).reshape(N,N)
    for i in range(N):
        a=E[i,0]
        b=int(a)
        for j in range(b):
            k=E[i,j+1]
            l=int(k)
            M[i,l]=1
    return(M)


def doubleconstellation(N,k):
    M=np.zeros(N*N).reshape(N,N)
    for i in range(k):
        for j in range(k):
            M[i,j+k]=1            
    for i in range(N-2*k):
        for j in range(k):
            M[i+2*k,j]=1
            M[j+k,i+2*k]=1
    return(M)



def constellationwithrandom(N,k,p):
    Edges=(2*(N-2*k)*k-k*k)*1.0
    M=np.zeros(N*N).reshape(N,N)
    for i in range(k):
        for j in range(k):
            M[i,j+k]=1            
    for i in range(N-2*k):
        for j in range(k):
            M[i+2*k,j]=1
            M[j+k,i+2*k]=1
    p=p-Edges*1.0/(N*N-N)
    Q = np.random.rand(N**2).reshape(N,N)
    for i in range(N):
        for j in range(N):
            if (i == j):
                M[i,j] = 0
            else:
                if (Q[i,j] < p):
                    M[i,j] = 1 #i connects to j
    return(M)


def constantdegree(N,k):
    M=zeros(N*N).reshape(N,N)
    for i in range(N):
        for j in range(k):
            m=np.random.randint(N)
            M[i,m]=1
    return(M)


def coordinates(A,m,d):
    X=A
    coords=np.zeros(d)
    remainder=np.zeros(d)
    for i in range(d):
        (X,coords[i])=divmod(X,m)
    return(coords)

def distance(A,B,m,d):
    coordsa = coordinates(A,m,d)
    #print(coordsa)
    coordsb = coordinates(B,m,d)
    #print(coordsb)
    vecdist=np.zeros(d)
    for i in range(d):
        vecdist[i]=(coordsa[i]-coordsb[i])**2
    dist=np.sum(vecdist)
    return(dist)

def localmatr(m,d,p,a):
    N=m**d
    M = np.random.rand(N**2).reshape(N,N)
    for i in range(N):
        for j in range(N):
            if (i == j):
                M[i,j] = 0
            else:
                if (M[i,j] < p*np.exp(-a*distance(i,j,m,d))):
                    #print(distance(i,j,m,d))
                    M[i,j] = 1 #i connects to j
                else:
                    M[i,j] = 0
    return M

    


def ranvec(N,n):
    vec=np.zeros(N)
    for i in range(n):
        a=np.random.randint(N)
        while(vec[a]==1):
            a=np.random.randint(N)
        vec[a]=1
    return(vec)

def random_initial_state(N,n_times,n):
    states = np.zeros(N*n_times).reshape(N,n_times) - 1
    vec = ranvec(N,n)
    for i in range(N):
        states[i,0] = -1  - vec[i]
    return(states)

def add_one_active(N,vec):
    i = np.random.randint(N)
    while(vec[i] == 1):
        i = np.random.randint(N)
    vec[i] = 1
    return(vec)

def vector_to_matrix(N,n_times,vec):
    states = np.zeros(N*n_times).reshape(N,n_times) - 1
    for i in range(N):
        states[i,0] = -1 - vec[i]
    return(states)


def spike_raster_plot(N, n_timesteps, stepsize, states):
    neuralData = []
    for i in range(N):
        neuralData.append([])
        for j in range(n_timesteps):
            if (states[i,j] == 0):
                neuralData[i].append(j*stepsize)
    return(neuralData)


def weight_matrix(n_neurons, edges, mu,sigma): 
   W_matrix = np.zeros(n_neurons**2).reshape(n_neurons, n_neurons)
   for i in range(n_neurons):
     neigh = int(edges[i,0])
     for j in range(neigh):
      W_matrix[i,j] = np.random.lognormal(mu, sigma)
   return(W_matrix)

def list_from_matrices(N, edges, syn, W):
  e_l = []
  s_l = []
  W_l = []
  for i in range(N):
    neigh = int(edges[i,0])
    e_l.append([])
    s_l.append([])
    W_l.append([])
    for j in range(neigh):
      e_l[i].append(edges[i,j+1])
      s_l[i].append(syn[i,j])
      W_l[i].append(W[i,j])
  return(e_l,s_l,W_l)

@jit(nopython=True)
def prepare_syndel_2(average, std, edges, n_neurons ):
    synaptic_del = np.zeros(n_neurons**2).reshape(n_neurons, n_neurons)
    for i in range(n_neurons):
      neigh = int(edges[i,0])
      for j in range(neigh):
        synaptic_del[i,j] = np.random.uniform(average - std, average + std)
    return(synaptic_del)

def active_set(N, n_activated):
  set_activated = np.zeros(n_activated)
  for i in range(n_activated):
    a = np.random.randint(N)
    for j in range(i):
      while (a == set_activated[j]):
        a = np.random.randint(N)
    set_activated[i] = a
  return(set_activated)    

def remove_one_active(set_activated):
    l = len(set_activated)
    new_set = np.zeros(l-1)
    for i in range(l-1):
        new_set[i] = set_activated[i]
    return(new_set)


@jit(nopython=True)
def lists_to_matrices_2(N, edges, W, syn_del):
    
    W_mat = np.zeros(N*N).reshape(N,N)
    syn_mat = np.zeros(N*N).reshape(N,N)

    for i in range(N):
        n_neighbors = int(edges[i,0])
        for j in range(n_neighbors):
            q = int(edges[i, j + 1])
            W_mat[q,i] = W[i, j]
            syn_mat[q,i] = syn_del[i,j]
    return(W_mat,syn_mat)   

def generate_network(params, network_type = 'ER'):
    if (network_type=='ER'):
        n_neurons = int(params['n_neurons'])
        p = params['ER connection probability']
        mu = params['mu']

        sigma = params['sigma']
        M = getRandomConnectivity(n_neurons,p)
        edges = matrixOfEdges(M,n_neurons)
        W_matrix = weight_matrix(n_neurons, edges, mu,sigma) 
        avdel = params['Average synaptic delay']  
        stddel = params['Standard deviation for synaptic delay']
        syn_del = prepare_syndel_2(avdel, stddel, edges, n_neurons ) #matrix of synaptic delays
    return(M, edges, W_matrix, syn_del)
