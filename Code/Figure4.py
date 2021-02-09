#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pystan
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from scipy import stats
from scipy.stats.stats import pearsonr
import pickle
from mpl_toolkits.mplot3d import Axes3D

FONT_SIZE = 8

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
plt.rcParams['pdf.fonttype'] = 42

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
plt.style.use('seaborn-white')

#%%
# load in required parameters

dat_RC = pd.read_csv('smFISH_RC.csv')

jtime_RC = dat_RC['Time'].values
AreaNormed_RC = dat_RC['AreaNormed'].values
CountsNr1d1_RC = dat_RC['Counts Nr1d1'].values
CountsCry1_RC = dat_RC['Counts Cry1'].values

dat_BC = pd.read_csv('smFISH_BC.csv')

jtime_BC = dat_BC['Time'].values
AreaNormed_BC = dat_BC['AreaNormed'].values
CountsBmal1_BC = dat_BC['Counts Bmal1'].values
CountsCry1_BC = dat_BC['Counts Cry1'].values

CountsCry1TOT = np.concatenate((CountsCry1_BC,CountsCry1_RC))
jtime_Cry1TOT = np.concatenate((jtime_BC,jtime_RC))

w = 2 * np.pi / 24 

sm = pickle.load(open('model_2.pkl', 'rb'))
fit = pickle.load(open('fit_2.pkl', 'rb')) 
    
a = fit.extract(permuted=True)

freq_scaleCry1 = np.mean(a['freq_scaleCry1'],axis=0)
freq_scaleNr1d1 = np.mean(a['freq_scaleNr1d1'],axis=0)
freq_scaleBmal1 = np.mean(a['freq_scaleBmal1'],axis=0)
beta_Cry1 = np.mean(a['beta_v_Cry1'],axis=0)
beta_Nr1d1 = np.mean(a['beta_v_Nr1d1'],axis=0)
beta_Bmal1 = np.mean(a['beta_v_Bmal1'],axis=0)


sm = pickle.load(open('model_4.pkl', 'rb'))
fit = pickle.load(open('fit_4.pkl', 'rb')) 
    
a = fit.extract(permuted=True)

burstalphaCry1 = np.mean(a['burstCry1'],axis=0)
burstalphaNr1d1 = np.mean(a['burstNr1d1'],axis=0)
burstalphaBmal1 = np.mean(a['burstBmal1'],axis=0)

mu_RC = np.mean(a['mu_vec_RC'],axis=0)#.reshape(-1,1)
Cov_RC = np.mean(a['cov_RC'],axis=0)

mu_BC = np.mean(a['mu_vec_BC'],axis=0)#.reshape(-1,1)
Cov_BC = np.mean(a['cov_BC'],axis=0)

Nr1d1_params, Cry1_params, Bmal1_params = pickle.load(open('FourierParams.pkl', 'rb'))

#%%

# calcualte covariance matrix K of \eta for simulations

K11 = Cov_RC[0,0]
K22 = Cov_RC[1,1]
K33 = Cov_BC[1,1]

K12 = Cov_RC[0,1]
K13 = Cov_BC[0,1]
  
A11 = K11**(-1)*(K12**2+(-1)*K11*K22)**(-1)*(K13**2+(-1)*K11*K33)**(-1)*((-1)*K12**2*K13**2+K11**2*K22*K33)
  
A12 = K12*(K12**2+(-1)*K11*K22)**(-1)
  
A13 = K13*(K13**2+(-1)*K11*K33)**(-1);
  
A22 = K11*((-1)*K12**2+K11*K22)**(-1);
  
A33 = K11*((-1)*K13**2+K11*K33)**(-1);

A23 = 0

A = [[A11,A12,A13],[A12,A22,A23],[A13,A23,A33]]

K = np.linalg.inv(A) 
    
#%%
# simulate model

def simulate_model(its):
    area = np.repeat(np.hstack((AreaNormed_BC,AreaNormed_RC)), its, axis=None)
    num_cells = len(area)
    
    eta = np.exp(np.random.multivariate_normal([burstalphaCry1,burstalphaNr1d1,burstalphaBmal1], K, size = num_cells))
    
    vargamm_Cry1 = np.zeros(np.size(area))
    vargamm_Nr1d1 = np.zeros(np.size(area))
    vargamm_Bmal1 = np.zeros(np.size(area))
    
    mu_vec_Cry1 = np.zeros(np.size(area))
    mu_vec_Nr1d1 = np.zeros(np.size(area))
    mu_vec_Bmal1 = np.zeros(np.size(area))
    
    t_vec = np.zeros(np.size(area))
    
    r1_vec = np.zeros(np.size(area))
    r2_vec = np.zeros(np.size(area))
    r3_vec = np.zeros(np.size(area))
    
    synthCry1 = np.zeros(np.size(area))
    synthNr1d1 = np.zeros(np.size(area))
    synthBmal1 = np.zeros(np.size(area))
    
    for j in range(num_cells):
        t = np.random.uniform(0,24)
        eta_curr = eta[j,:]
        r1 = freq_scaleCry1*(Cry1_params[0]/2+Cry1_params[1]*np.cos(t*w-Cry1_params[3])+Cry1_params[2]*np.cos(2*t*w-Cry1_params[4]))
        r2 = freq_scaleNr1d1*(Nr1d1_params[0]/2+Nr1d1_params[1]*np.cos(t*w-Nr1d1_params[3])+Nr1d1_params[2]*np.cos(2*t*w-Nr1d1_params[4]))
        r3 = freq_scaleBmal1*(Bmal1_params[0]/2+Bmal1_params[1]*np.cos(t*w-Bmal1_params[3])+Bmal1_params[2]*np.cos(2*t*w-Bmal1_params[4]))
        mu_1 = eta_curr[0]*(area[j])**beta_Cry1*r1  
        mu_2 = eta_curr[1]*(area[j])**beta_Nr1d1*r2
        mu_3 = eta_curr[2]*(area[j])**beta_Bmal1*r3 
        mu_vec_Cry1[j] = mu_1
        mu_vec_Nr1d1[j] = mu_2
        mu_vec_Bmal1[j] = mu_3
        r1_vec[j] = r1
        r2_vec[j] = r2
        r3_vec[j] = r3
        vargamm_Cry1[j] = mu_1**2/r1
        vargamm_Nr1d1[j] = mu_2**2/r2
        vargamm_Bmal1[j] = mu_3**2/r3
        m1 = mu_1    
        p1 = r1/(m1+r1)
        m2 = mu_2    
        p2 = r2/(m2+r2)
        m3 = mu_3    
        p3 = r3/(m3+r3)
        synthCry1[j] = np.random.negative_binomial(r1, p1)
        synthNr1d1[j] = np.random.negative_binomial(r2, p2)
        synthBmal1[j] = np.random.negative_binomial(r3, p3)
    return synthCry1, synthNr1d1, synthBmal1, area, r1_vec, r2_vec, r3_vec, eta, mu_vec_Cry1, mu_vec_Nr1d1, mu_vec_Bmal1, vargamm_Cry1, vargamm_Nr1d1, vargamm_Bmal1
    
synthCry1, synthNr1d1, synthBmal1, area, r1_vec, r2_vec, r3_vec, eta, mu_vec_Cry1, mu_vec_Nr1d1, mu_vec_Bmal1, vargamm_Cry1, vargamm_Nr1d1, vargamm_Bmal1 = simulate_model(100)


def component_variance(X1,X2,X3):
    var_1 = np.var(X1*np.mean(X2)*np.mean(X3))
    var_X1 = np.var(X1)
    var_vec = np.zeros(len(X1))
    
    mean_X2 = np.mean(X2)
    mean_X3 = np.mean(X3
                      )
    for i, (X1_curr,X2_curr,X3_curr) in enumerate(zip(X1,X2,X3)):
        var_vec[i] = (X2_curr*mean_X3)**2*var_X1
        
    var_2 = np.mean(var_vec)
    
    for i, (X1_curr,X2_curr,X3_curr) in enumerate(zip(X1,X2,X3)):
        var_vec[i] = (mean_X2*X3_curr)**2*var_X1    
    
    var_3 = np.mean(var_vec)  
    
    for i, (X1_curr,X2_curr,X3_curr) in enumerate(zip(X1,X2,X3)):
        var_vec[i] = (X2_curr*X3_curr)**2*var_X1 
    
    var_4 = np.mean(var_vec)
    
    return var_1, var_2, var_3, var_4

var_1, var_2, var_3, var_4 = component_variance(eta[:,0],r1_vec,area**(beta_Cry1))
var_eta_Cry1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)
var_1, var_2, var_3, var_4 = component_variance(area**(beta_Cry1),r1_vec,eta[:,0])
var_area_Cry1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)
var_1, var_2, var_3, var_4 = component_variance(r1_vec,area**(beta_Cry1),eta[:,0])
var_t_Cry1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)

var_1, var_2, var_3, var_4 = component_variance(eta[:,1],r2_vec,area**(beta_Nr1d1))
var_eta_Nr1d1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)
var_1, var_2, var_3, var_4 = component_variance(area**(beta_Nr1d1),r2_vec,eta[:,1])
var_area_Nr1d1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)
var_1, var_2, var_3, var_4 = component_variance(r2_vec,area**(beta_Nr1d1),eta[:,1])
var_t_Nr1d1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)

var_1, var_2, var_3, var_4 = component_variance(eta[:,2],r3_vec,area**(beta_Bmal1))
var_eta_Bmal1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)
var_1, var_2, var_3, var_4 = component_variance(area**(beta_Bmal1),r3_vec,eta[:,2])
var_area_Bmal1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)
var_1, var_2, var_3, var_4 = component_variance(r3_vec,area**(beta_Bmal1),eta[:,2])
var_t_Bmal1 = 1/3*(var_1+0.5*(var_2+var_3)+var_4)

var_burst_Cry1 = np.mean(vargamm_Cry1)
var_poiss_Cry1 = np.mean(mu_vec_Cry1)
var_tot_Cry1 = var_eta_Cry1+var_area_Cry1+var_t_Cry1+var_burst_Cry1+var_poiss_Cry1
var_tot_synth_Cry1 = np.var(synthCry1)
mean_tot_synth_Cry1 = np.mean(synthCry1)

var_burst_Nr1d1 = np.mean(vargamm_Nr1d1)
var_poiss_Nr1d1 = np.mean(mu_vec_Nr1d1)
var_tot_Nr1d1 = var_eta_Nr1d1+var_area_Nr1d1+var_t_Nr1d1+var_burst_Nr1d1+var_poiss_Nr1d1
var_tot_synth_Nr1d1 = np.var(synthNr1d1)
mean_tot_synth_Nr1d1 = np.mean(synthNr1d1)

var_burst_Bmal1 = np.mean(vargamm_Bmal1)
var_poiss_Bmal1 = np.mean(mu_vec_Bmal1)
var_tot_Bmal1 = var_eta_Bmal1+var_area_Bmal1+var_t_Bmal1+var_burst_Bmal1+var_poiss_Bmal1
var_tot_synth_Bmal1 = np.var(synthBmal1)
mean_tot_synth_Bmal1 = np.mean(synthBmal1)

#%%

def GetTimeVar(num_cells):    
    its = 10000
    area = np.hstack((AreaNormed_BC,AreaNormed_RC))
    eta = np.exp(np.random.multivariate_normal([burstalphaCry1,burstalphaNr1d1,burstalphaBmal1], K, size = len(area)))  
        
    mu_vec_Cry1 = np.zeros(its)
    mu_vec_Nr1d1 = np.zeros(its)
    mu_vec_Bmal1 = np.zeros(its)
    
    t_vec = np.zeros(its)
    
    Y_Cry1 = np.zeros(its)
    Y_Nr1d1 = np.zeros(its)
    Y_Bmal1 = np.zeros(its)
    
    mean_area_Cry1 = np.mean(area**beta_Cry1)
    mean_area_Nr1d1 = np.mean(area**beta_Nr1d1)
    mean_area_Bmal1 = np.mean(area**beta_Bmal1)
    
    mean_eta_Cry1 = np.mean(eta[:,0])
    mean_eta_Nr1d1 = np.mean(eta[:,1])
    mean_eta_Bmal1 = np.mean(eta[:,2])
    
    synthCry1 = np.zeros(num_cells)
    synthNr1d1 = np.zeros(num_cells)
    synthBmal1 = np.zeros(num_cells)
    
    for i in range(its):
        print(i)
        t = np.random.uniform(0,24)
        r1 = freq_scaleCry1*(Cry1_params[0]/2+Cry1_params[1]*np.cos(t*w-Cry1_params[3])+Cry1_params[2]*np.cos(2*t*w-Cry1_params[4]))
        r2 = freq_scaleNr1d1*(Nr1d1_params[0]/2+Nr1d1_params[1]*np.cos(t*w-Nr1d1_params[3])+Nr1d1_params[2]*np.cos(2*t*w-Nr1d1_params[4]))
        r3 = freq_scaleBmal1*(Bmal1_params[0]/2+Bmal1_params[1]*np.cos(t*w-Bmal1_params[3])+Bmal1_params[2]*np.cos(2*t*w-Bmal1_params[4]))
        for j in range(num_cells):   
    #         randomly sample from area and eta list        
            eta_curr = eta[np.random.randint(0,len(eta)),:]
            area_curr = area[np.random.randint(0,len(area))]
            mu_1 = eta_curr[0]*(area_curr)**beta_Cry1*r1  
            mu_2 = eta_curr[1]*(area_curr)**beta_Nr1d1*r2
            mu_3 = eta_curr[2]*(area_curr)**beta_Bmal1*r3 
            m1 = mu_1    
            p1 = r1/(m1+r1)
            m2 = mu_2    
            p2 = r2/(m2+r2)
            m3 = mu_3    
            p3 = r3/(m3+r3)
            synthCry1[j] = np.random.negative_binomial(r1, p1)
            synthNr1d1[j] = np.random.negative_binomial(r2, p2)
            synthBmal1[j] = np.random.negative_binomial(r3, p3)        
        t_vec[i] = t
        Y_Cry1[i] = np.mean(synthCry1)
        Y_Nr1d1[i] = np.mean(synthNr1d1)
        Y_Bmal1[i] = np.mean(synthBmal1)
        mu_vec_Cry1[i] = mean_area_Cry1*mean_eta_Cry1*r1
        mu_vec_Nr1d1[i] = mean_area_Nr1d1*mean_eta_Nr1d1*r2
        mu_vec_Bmal1[i] = mean_area_Bmal1*mean_eta_Bmal1*r3
    
    PercentCry1 = 100*np.var(mu_vec_Cry1)/np.var(Y_Cry1)
    PercentNr1d1 = 100*np.var(mu_vec_Nr1d1)/np.var(Y_Nr1d1)
    PercentBmal1 = 100*np.var(mu_vec_Bmal1)/np.var(Y_Bmal1)
    
    return PercentCry1, PercentNr1d1, PercentBmal1

num_cell_list = [1,10,25,50,100,500,1000]
num_cells_vec = np.array(num_cell_list)

PercentCry1_vec = np.zeros(len(num_cells_vec))
PercentNr1d1_vec = np.zeros(len(num_cells_vec))
PercentBmal1_vec = np.zeros(len(num_cells_vec))

for i in range(len(num_cells_vec)):
    PercentCry1_vec[i], PercentNr1d1_vec[i], PercentBmal1_vec[i] = GetTimeVar(num_cells_vec[i])

#%%

def TimeVaryingMean(t_vec):
    
    its = 10
    area = np.repeat(np.hstack((AreaNormed_BC,AreaNormed_RC)), its, axis=None)
    num_cells = len(area)
    
    eta = np.exp(np.random.multivariate_normal([burstalphaCry1,burstalphaNr1d1,burstalphaBmal1], K, size = num_cells))
    
    mean_area_Cry1 = np.mean(area**beta_Cry1)
    mean_area_Nr1d1 = np.mean(area**beta_Nr1d1)
    mean_area_Bmal1 = np.mean(area**beta_Bmal1)
    
    mean_eta_Cry1 = np.mean(eta[:,0])
    mean_eta_Nr1d1 = np.mean(eta[:,1])
    mean_eta_Bmal1 = np.mean(eta[:,2])
    
    mu_vec_Cry1 = np.zeros(len(t_vec))
    mu_vec_Nr1d1 = np.zeros(len(t_vec))
    mu_vec_Bmal1 = np.zeros(len(t_vec))
    
    for i in range(len(t_vec)):
        t = t_vec[i]
        r1 = freq_scaleCry1*(Cry1_params[0]/2+Cry1_params[1]*np.cos(t*w-Cry1_params[3])+Cry1_params[2]*np.cos(2*t*w-Cry1_params[4]))
        r2 = freq_scaleNr1d1*(Nr1d1_params[0]/2+Nr1d1_params[1]*np.cos(t*w-Nr1d1_params[3])+Nr1d1_params[2]*np.cos(2*t*w-Nr1d1_params[4]))
        r3 = freq_scaleBmal1*(Bmal1_params[0]/2+Bmal1_params[1]*np.cos(t*w-Bmal1_params[3])+Bmal1_params[2]*np.cos(2*t*w-Bmal1_params[4]))
        mu_vec_Cry1[i] = mean_area_Cry1*mean_eta_Cry1*r1
        mu_vec_Nr1d1[i] = mean_area_Nr1d1*mean_eta_Nr1d1*r2
        mu_vec_Bmal1[i] = mean_area_Bmal1*mean_eta_Bmal1*r3
    
    return mu_vec_Cry1, mu_vec_Nr1d1, mu_vec_Bmal1

mu_vec_Cry1, mu_vec_Nr1d1, mu_vec_Bmal1 = TimeVaryingMean(np.linspace(0,24,48))    

#%%

synthCry1_dens, synthNr1d1_dens, synthBmal1_dens, = simulate_model(10)[:3]

x = synthCry1_dens
y = synthNr1d1_dens
z = synthBmal1_dens
xyz = np.vstack([x,y,z])

kde = stats.gaussian_kde(xyz)

xvec = range(150)
yvec = range(150)
zvec = range(150)  

kde_tot = np.zeros((len(xvec),len(yvec),len(zvec)))  
val = np.zeros(3)

for i in range(len(xvec)):
    print(i)
    for j in range(len(yvec)):
        for k in range(len(zvec)):
            val[0] = xvec[i]
            val[1] = yvec[j]
            val[2] = zvec[k]
            kde_tot[i,j,k] = kde(val)
            
          
kde_tot_normed = kde_tot/np.sum(kde_tot)


# find region of highest density

trialvec = np.linspace(0,np.amax(kde_tot_normed),100)
prop_vec = np.zeros(len(trialvec))

for i in range(len(trialvec)):
    prop_vec[i] = np.sum(kde_tot_normed[kde_tot_normed>trialvec[i]])
  
ind = np.argmin(abs(prop_vec-0.80))
kde_tot_filt = kde_tot_normed>trialvec[ind]


X, Y = np.meshgrid(xvec,yvec)

Zfirst = np.zeros(X.shape)
Zlast = np.zeros(X.shape)

for i in range(len(xvec)):
    print(i)
    for j in range(len(yvec)):
        z_curr = kde_tot_filt[i,j,:]
        Zfirst[i,j] = z_curr.argmax()
        z_curr_r = np.flip(z_curr,axis=0)
        if z_curr_r.argmax()!=0:
            Zlast[i,j] = zvec[-1]-z_curr_r.argmax()#(logical_not(z_curr)).argmax()

mpl_fig = plt.figure(figsize=(12/2.54,6/2.54))
ax = mpl_fig.add_subplot(121)


ind = 1
width = 0.35       

mean2_tot_synth_Nr1d1 = mean_tot_synth_Nr1d1**2


p1 = ax.bar(ind, var_eta_Nr1d1/mean2_tot_synth_Nr1d1, width,label='Other extrinsic',color='lightblue')
p2 = ax.bar(ind, var_area_Nr1d1/mean2_tot_synth_Nr1d1, width, 
             bottom=var_eta_Nr1d1/mean2_tot_synth_Nr1d1,label='Area',color='cyan')
p3 = ax.bar(ind, var_t_Nr1d1/mean2_tot_synth_Nr1d1, width,
             bottom=var_eta_Nr1d1/mean2_tot_synth_Nr1d1+var_area_Nr1d1/mean2_tot_synth_Nr1d1,label='Time',color='lightgreen')

p4 = ax.bar(ind, var_burst_Nr1d1/mean2_tot_synth_Nr1d1, width, 
             bottom=var_eta_Nr1d1/mean2_tot_synth_Nr1d1+var_area_Nr1d1/mean2_tot_synth_Nr1d1+var_t_Nr1d1/mean2_tot_synth_Nr1d1,label='Burst',color='orange')
p5 = ax.bar(ind, var_poiss_Nr1d1/mean2_tot_synth_Nr1d1, width, 
             bottom=var_eta_Nr1d1/mean2_tot_synth_Nr1d1+var_area_Nr1d1/mean2_tot_synth_Nr1d1+var_t_Nr1d1/mean2_tot_synth_Nr1d1+var_burst_Nr1d1/mean2_tot_synth_Nr1d1,label='Poiss',color='red')


ind = 2

mean2_tot_synth_Cry1 = mean_tot_synth_Cry1**2

p1 = ax.bar(ind, var_eta_Cry1/mean2_tot_synth_Cry1, width,color='lightblue')
p2 = ax.bar(ind, var_area_Cry1/mean2_tot_synth_Cry1, width, 
             bottom=var_eta_Cry1/mean2_tot_synth_Cry1,color='cyan')
p3 = ax.bar(ind, var_t_Cry1/mean2_tot_synth_Cry1, width,
             bottom=var_eta_Cry1/mean2_tot_synth_Cry1+var_area_Cry1/mean2_tot_synth_Cry1,color='lightgreen')

p4 = ax.bar(ind, var_burst_Cry1/mean2_tot_synth_Cry1, width, 
             bottom=var_eta_Cry1/mean2_tot_synth_Cry1+var_area_Cry1/mean2_tot_synth_Cry1+var_t_Cry1/mean2_tot_synth_Cry1,color='orange')
p5 = ax.bar(ind, var_poiss_Cry1/mean2_tot_synth_Cry1, width, 
             bottom=var_eta_Cry1/mean2_tot_synth_Cry1+var_area_Cry1/mean2_tot_synth_Cry1+var_t_Cry1/mean2_tot_synth_Cry1+var_burst_Cry1/mean2_tot_synth_Cry1,color='red')


ind = 3

mean2_tot_synth_Bmal1 = mean_tot_synth_Bmal1**2

p1 = ax.bar(ind, var_eta_Bmal1/mean2_tot_synth_Bmal1, width,color='lightblue')
p2 = ax.bar(ind, var_area_Bmal1/mean2_tot_synth_Bmal1, width, 
             bottom=var_eta_Bmal1/mean2_tot_synth_Bmal1,color='cyan')
p3 = ax.bar(ind, var_t_Bmal1/mean2_tot_synth_Bmal1, width,
             bottom=var_eta_Bmal1/mean2_tot_synth_Bmal1+var_area_Bmal1/mean2_tot_synth_Bmal1,color='lightgreen')

p4 = ax.bar(ind, var_burst_Bmal1/mean2_tot_synth_Bmal1, width, 
             bottom=var_eta_Bmal1/mean2_tot_synth_Bmal1+var_area_Bmal1/mean2_tot_synth_Bmal1+var_t_Bmal1/mean2_tot_synth_Bmal1,color='orange')
p5 = ax.bar(ind, var_poiss_Bmal1/mean2_tot_synth_Bmal1, width, 
             bottom=var_eta_Bmal1/mean2_tot_synth_Bmal1+var_area_Bmal1/mean2_tot_synth_Bmal1+var_t_Bmal1/mean2_tot_synth_Bmal1+var_burst_Bmal1/mean2_tot_synth_Bmal1,color='red')




legend = plt.legend(loc='upper center', shadow=True)
ax.set_ylabel(r'$\eta^2$')


ax.set_xticks(np.arange(3) + 1)
ax.set_xticklabels(('Nr1d1', 'Cry1', 'Bmal1'))

axes = plt.gca()
a = axes.get_xlim()
b = axes.get_ylim()
plt.text(a[0]-0.1*(a[1]-a[0]),b[1]+0.03*(b[1]-b[0]), 'A',fontsize=12)


ax = mpl_fig.add_subplot(122)

plt.scatter(range(len(num_cells_vec)),PercentCry1_vec,color='green',label='Cry1',s=2) 
plt.scatter(range(len(num_cells_vec)),PercentNr1d1_vec,color='blue',label='Nr1d1',s=2)   
plt.scatter(range(len(num_cells_vec)),PercentBmal1_vec,color='red',label='Bmal1',s=2)
xL=[str(i) for i in num_cell_list]
plt.xticks(range(len(num_cells_vec)), xL)  
legend = plt.legend(loc='upper left', shadow=True)  
plt.xlabel('Number of cells') 
plt.ylabel('% variance: time') 

plt.tight_layout()

axes = plt.gca()
a = axes.get_xlim()
b = axes.get_ylim()
plt.text(a[0]-0.1*(a[1]-a[0]),b[1]+0.03*(b[1]-b[0]), 'B',fontsize=12)


mpl_fig = plt.figure(figsize=(18/2.54,9/2.54))
ax = plt.subplot(121, projection='3d', facecolor='white')


plt.plot(mu_vec_Cry1,mu_vec_Nr1d1,mu_vec_Bmal1)

ax.view_init(elev=44., azim=27)


ax.set_xlabel('mean Cry1 [mRNA/cell]')
ax.set_ylabel('mean Nr1d1 [mRNA/cell]')
ax.set_zlabel('mean Bmal1 [mRNA/cell]')

ax = plt.subplot(122, projection='3d', facecolor='white')

x = X[Zlast!=0]
y = Y[Zlast!=0]
z = Zlast[Zlast!=0]

alpha = 0.3

ax.scatter(x, y, z,s=1,alpha = alpha,color='blue',marker=',')

x = X[Zfirst!=0]
y = Y[Zfirst!=0]
z = Zfirst[Zfirst!=0]

ax.scatter(x, y, z,s=1,alpha = alpha,color='blue',marker=',')     

ax.view_init(elev=44., azim=27) 

ax.set_xlabel('Cry1 mRNA/cell')
ax.set_ylabel('Nr1d1 mRNA/cell')
ax.set_zlabel('Bmal1 mRNA/cell')
