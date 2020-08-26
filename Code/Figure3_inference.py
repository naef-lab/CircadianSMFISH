#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pystan
import stan_utility
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from scipy import stats
from scipy.stats.stats import pearsonr
import pickle

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

### load data ###
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
AreaCry1TOT = np.concatenate((AreaNormed_BC,AreaNormed_RC))

w = 2 * np.pi / 24 

TimevecRC = np.array([21.,25.,29.,33.,37.,41.]).reshape(-1,1)
TimevecBC = np.array([17.,21.,25.,29.,33.,37.,41.]).reshape(-1,1)

#%%
# load Fourier parameters
Nr1d1_params, Cry1_params, Bmal1_params = pickle.load(open('FourierParams.pkl', 'rb'))

#%%

# MODEL 1

model1 = """
data{
    int<lower=1> N1;
    int CountsNr1d1_RC[N1];
    int CountsCry1_RC[N1];    
    real jtime_RC[N1];
    real AreaNormed_RC[N1];
    int<lower=1> N2;
    int CountsBmal1_BC[N2];
    int CountsCry1_BC[N2];    
    real jtime_BC[N2];
    real AreaNormed_BC[N2];
    int <lower=1> N_f;
    real Nr1d1_params[N_f];
    real Cry1_params[N_f];
    real Bmal1_params[N_f];
    real w;
}
parameters{
    real<lower=0> freq_scaleCry1;
    real<lower=0> freq_scaleNr1d1; 
    real<lower=0> freq_scaleBmal1; 
    real<lower=0> burstCry1;
    real<lower=0> burstNr1d1;
    real<lower=0> burstBmal1;
}
model{
    vector[N1] mu1;
    vector[N1] mu2;
    vector[N2] mu3;
    vector[N2] mu4;
    vector[N1] r1;
    vector[N1] r2;
    vector[N2] r3;
    vector[N2] r4;
    real b1;
    real b2;
    real b3;
    real b4;
    real f1;
    real f2;
    real f3;
    real f4;
    for ( i in 1:N1 ) {
        b1 = burstCry1;
        b2 = burstNr1d1;
        f1 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_RC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_RC[i]*w-Cry1_params[5]));
        f2 = freq_scaleNr1d1*(Nr1d1_params[1]/2+Nr1d1_params[2]*cos(jtime_RC[i]*w-Nr1d1_params[4])+Nr1d1_params[3]*cos(2*jtime_RC[i]*w-Nr1d1_params[5]));
        mu1[i] = b1*f1;
        mu2[i] = b2*f2;
        r1[i] = f1;
        r2[i] = f2;
    } 
    for ( i in 1:N2 ) {
        b3 = burstCry1;
        b4 = burstBmal1;
        f3 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_BC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_BC[i]*w-Cry1_params[5]));
        f4 = freq_scaleBmal1*(Bmal1_params[1]/2+Bmal1_params[2]*cos(jtime_BC[i]*w-Bmal1_params[4])+Bmal1_params[3]*cos(2*jtime_BC[i]*w-Bmal1_params[5]));
        mu3[i] = b3*f3;
        mu4[i] = b4*f4;
        r3[i] = f3;
        r4[i] = f4;
    } 
    CountsCry1_RC ~ neg_binomial_2( mu1 , r1 );
    CountsNr1d1_RC ~ neg_binomial_2( mu2 , r2 );
    CountsCry1_BC ~ neg_binomial_2( mu3 , r3 );
    CountsBmal1_BC ~ neg_binomial_2( mu4 , r4 );
    freq_scaleCry1~ normal(0,100);
    freq_scaleNr1d1~ normal(0,100);
    freq_scaleBmal1~ normal(0,100);
    burstCry1~ normal(0,100);
    burstNr1d1~ normal(0,100);
    burstBmal1~ normal(0,100);
}
generated quantities{ 
    vector[N1] log_lik1;
    vector[N1] log_lik2;
    vector[N2] log_lik3;
    vector[N2] log_lik4;
    real mu1;
    real mu2;
    real mu3;
    real mu4;
    real r1;
    real r2;
    real r3;
    real r4;
    real b1;
    real b2;
    real b3;
    real b4;
    real f1;
    real f2;
    real f3;
    real f4;
    for ( i in 1:N1 ) {
        b1 = burstCry1;
        b2 = burstNr1d1;
        f1 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_RC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_RC[i]*w-Cry1_params[5]));
        f2 = freq_scaleNr1d1*(Nr1d1_params[1]/2+Nr1d1_params[2]*cos(jtime_RC[i]*w-Nr1d1_params[4])+Nr1d1_params[3]*cos(2*jtime_RC[i]*w-Nr1d1_params[5]));
        mu1 = b1*f1;
        mu2 = b2*f2;
        r1 = f1;
        r2 = f2;
        log_lik1[i] = neg_binomial_2_lpmf(CountsCry1_RC[i] | mu1 , r1);
        log_lik2[i] = neg_binomial_2_lpmf(CountsNr1d1_RC[i] | mu2 , r2);
    } 
    for ( i in 1:N2 ) {
        b3 = burstCry1;
        b4 = burstBmal1;
        f3 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_BC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_BC[i]*w-Cry1_params[5]));
        f4 = freq_scaleBmal1*(Bmal1_params[1]/2+Bmal1_params[2]*cos(jtime_BC[i]*w-Bmal1_params[4])+Bmal1_params[3]*cos(2*jtime_BC[i]*w-Bmal1_params[5]));
        mu3 = b3*f3;
        mu4 = b4*f4;
        r3 = f3;
        r4 = f4;
        log_lik3[i] = neg_binomial_2_lpmf(CountsCry1_BC[i] | mu3 , r3);
        log_lik4[i] = neg_binomial_2_lpmf(CountsBmal1_BC[i] | mu4 , r4);
    } 
    

}
 

"""

#%%


dat = {
    'N1' : len(CountsNr1d1_RC),
    'CountsNr1d1_RC' : CountsNr1d1_RC,
    'CountsCry1_RC' : CountsCry1_RC,   
    'jtime_RC' : jtime_RC, 
    'AreaNormed_RC' : AreaNormed_RC, 
    'N2' : len(CountsBmal1_BC),
    'CountsBmal1_BC' : CountsBmal1_BC,
    'CountsCry1_BC' : CountsCry1_BC,    
    'jtime_BC' : jtime_BC,
    'AreaNormed_BC' : AreaNormed_BC,
    'N_f' : len(Nr1d1_params),
    'Nr1d1_params' : Nr1d1_params,
    'Cry1_params' : Cry1_params,
    'Bmal1_params' : Bmal1_params,
    'w' : w
}


sm = pystan.StanModel(model_code=model1)

fit = sm.sampling(data=dat, seed=194838, iter=2000, chains=4, control=dict(adapt_delta=0.95))

with open('model_1.pkl', 'wb') as f:
    pickle.dump(sm, f)
    
with open('fit_1.pkl', 'wb') as g:
    pickle.dump(fit, g)
         
#%%

# MODEL 2

model1 = """
data{
    int<lower=1> N1;
    int CountsNr1d1_RC[N1];
    int CountsCry1_RC[N1];    
    real jtime_RC[N1];
    real AreaNormed_RC[N1];
    int<lower=1> N2;
    int CountsBmal1_BC[N2];
    int CountsCry1_BC[N2];    
    real jtime_BC[N2];
    real AreaNormed_BC[N2];
    int <lower=1> N_f;
    real Nr1d1_params[N_f];
    real Cry1_params[N_f];
    real Bmal1_params[N_f];
    real w;
}
parameters{
    real<lower=0> beta_v_Cry1;
    real<lower=0> beta_v_Nr1d1;
    real<lower=0> beta_v_Bmal1;
    real<lower=0> freq_scaleCry1;
    real<lower=0> freq_scaleNr1d1; 
    real<lower=0> freq_scaleBmal1; 
    real<lower=0> burstCry1;
    real<lower=0> burstNr1d1;
    real<lower=0> burstBmal1;
}
model{
    vector[N1] mu1;
    vector[N1] mu2;
    vector[N2] mu3;
    vector[N2] mu4;
    vector[N1] r1;
    vector[N1] r2;
    vector[N2] r3;
    vector[N2] r4;
    real b1;
    real b2;
    real b3;
    real b4;
    real f1;
    real f2;
    real f3;
    real f4;
    for ( i in 1:N1 ) {
        b1 = burstCry1*(AreaNormed_RC[i])^beta_v_Cry1;
        b2 = burstNr1d1*(AreaNormed_RC[i])^beta_v_Nr1d1;
        f1 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_RC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_RC[i]*w-Cry1_params[5]));
        f2 = freq_scaleNr1d1*(Nr1d1_params[1]/2+Nr1d1_params[2]*cos(jtime_RC[i]*w-Nr1d1_params[4])+Nr1d1_params[3]*cos(2*jtime_RC[i]*w-Nr1d1_params[5]));
        mu1[i] = b1*f1;
        mu2[i] = b2*f2;
        r1[i] = f1;
        r2[i] = f2;
    } 
    for ( i in 1:N2 ) {
        b3 = burstCry1*(AreaNormed_BC[i])^beta_v_Cry1;
        b4 = burstBmal1*(AreaNormed_BC[i])^beta_v_Bmal1;
        f3 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_BC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_BC[i]*w-Cry1_params[5]));
        f4 = freq_scaleBmal1*(Bmal1_params[1]/2+Bmal1_params[2]*cos(jtime_BC[i]*w-Bmal1_params[4])+Bmal1_params[3]*cos(2*jtime_BC[i]*w-Bmal1_params[5]));
        mu3[i] = b3*f3;
        mu4[i] = b4*f4;
        r3[i] = f3;
        r4[i] = f4;
    } 
    CountsCry1_RC ~ neg_binomial_2( mu1 , r1 );
    CountsNr1d1_RC ~ neg_binomial_2( mu2 , r2 );
    CountsCry1_BC ~ neg_binomial_2( mu3 , r3 );
    CountsBmal1_BC ~ neg_binomial_2( mu4 , r4 );
    freq_scaleCry1~ normal(0,100);
    freq_scaleNr1d1~ normal(0,100);
    freq_scaleBmal1~ normal(0,100);
    burstCry1~ normal(0,100);
    burstNr1d1~ normal(0,100);
    burstBmal1~ normal(0,100);
    beta_v_Cry1~ normal(0,100);
    beta_v_Nr1d1~ normal(0,100);
    beta_v_Bmal1~ normal(0,100);
}
generated quantities{ 
    vector[N1] log_lik1;
    vector[N1] log_lik2;
    vector[N2] log_lik3;
    vector[N2] log_lik4;
    real mu1;
    real mu2;
    real mu3;
    real mu4;
    real r1;
    real r2;
    real r3;
    real r4;
    real b1;
    real b2;
    real b3;
    real b4;
    real f1;
    real f2;
    real f3;
    real f4;
    for ( i in 1:N1 ) {
        b1 = burstCry1*(AreaNormed_RC[i])^beta_v_Cry1;
        b2 = burstNr1d1*(AreaNormed_RC[i])^beta_v_Nr1d1;
        f1 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_RC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_RC[i]*w-Cry1_params[5]));
        f2 = freq_scaleNr1d1*(Nr1d1_params[1]/2+Nr1d1_params[2]*cos(jtime_RC[i]*w-Nr1d1_params[4])+Nr1d1_params[3]*cos(2*jtime_RC[i]*w-Nr1d1_params[5]));
        mu1 = b1*f1;
        mu2 = b2*f2;
        r1 = f1;
        r2 = f2;
        log_lik1[i] = neg_binomial_2_lpmf(CountsCry1_RC[i] | mu1 , r1);
        log_lik2[i] = neg_binomial_2_lpmf(CountsNr1d1_RC[i] | mu2 , r2);
    } 
    for ( i in 1:N2 ) {
        b3 = burstCry1*(AreaNormed_BC[i])^beta_v_Cry1;
        b4 = burstBmal1*(AreaNormed_BC[i])^beta_v_Bmal1;
        f3 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_BC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_BC[i]*w-Cry1_params[5]));
        f4 = freq_scaleBmal1*(Bmal1_params[1]/2+Bmal1_params[2]*cos(jtime_BC[i]*w-Bmal1_params[4])+Bmal1_params[3]*cos(2*jtime_BC[i]*w-Bmal1_params[5]));
        mu3 = b3*f3;
        mu4 = b4*f4;
        r3 = f3;
        r4 = f4;
        log_lik3[i] = neg_binomial_2_lpmf(CountsCry1_BC[i] | mu3 , r3);
        log_lik4[i] = neg_binomial_2_lpmf(CountsBmal1_BC[i] | mu4 , r4);
    } 
    

}    

"""

#%%

dat = {
    'N1' : len(CountsNr1d1_RC),
    'CountsNr1d1_RC' : CountsNr1d1_RC,
    'CountsCry1_RC' : CountsCry1_RC,   
    'jtime_RC' : jtime_RC, 
    'AreaNormed_RC' : AreaNormed_RC, 
    'N2' : len(CountsBmal1_BC),
    'CountsBmal1_BC' : CountsBmal1_BC,
    'CountsCry1_BC' : CountsCry1_BC,    
    'jtime_BC' : jtime_BC,
    'AreaNormed_BC' : AreaNormed_BC,
    'N_f' : len(Nr1d1_params),
    'Nr1d1_params' : Nr1d1_params,
    'Cry1_params' : Cry1_params,
    'Bmal1_params' : Bmal1_params,
    'w' : w
}

sm = pystan.StanModel(model_code=model1)

fit = sm.sampling(data=dat, seed=194838, iter=2000, chains=4, control=dict(adapt_delta=0.95))

with open('model_2.pkl', 'wb') as f:
    pickle.dump(sm, f)
    
with open('fit_2.pkl', 'wb') as g:
    pickle.dump(fit, g)
    

#%%
# MODEL 3 - phase noise

model1 = """
data{
    int<lower=1> N1;
    int CountsNr1d1_RC[N1];
    int CountsCry1_RC[N1];    
    real jtime_RC[N1];
    real AreaNormed_RC[N1];
    int<lower=1> N2;
    int CountsBmal1_BC[N2];
    int CountsCry1_BC[N2];    
    real jtime_BC[N2];
    real AreaNormed_BC[N2];
    real kappa;
    real<lower=0> beta_v_Cry1;
    real<lower=0> beta_v_Nr1d1;
    real<lower=0> beta_v_Bmal1;
    real<lower=0> freq_scaleCry1;
    real<lower=0> freq_scaleNr1d1; 
    real<lower=0> freq_scaleBmal1; 
    real<lower=0> burstCry1;
    real<lower=0> burstNr1d1;
    real<lower=0> burstBmal1;
    int <lower=1> N_f;
    real Nr1d1_params[N_f];
    real Cry1_params[N_f];
    real Bmal1_params[N_f];
    real w;
}
parameters{
    real phi_i_add_RC[N1];
    real phi_i_add_BC[N2];
    real<lower=0,upper=2> amp_scal;
}
model{
    vector[N1] mu1;
    vector[N1] mu2;
    vector[N2] mu3;
    vector[N2] mu4;
    vector[N1] r1;
    vector[N1] r2;
    vector[N2] r3;
    vector[N2] r4;
    real b1;
    real b2;
    real b3;
    real b4;
    real f1;
    real f2;
    real f3;
    real f4;
    for ( i in 1:N1 ) {
        b1 = burstCry1*(AreaNormed_RC[i])^beta_v_Cry1;
        b2 = burstNr1d1*(AreaNormed_RC[i])^beta_v_Nr1d1;
        f1 = freq_scaleCry1*(Cry1_params[1]/2+amp_scal*Cry1_params[2]*cos(jtime_RC[i]*w-Cry1_params[4]-phi_i_add_RC[i])+amp_scal*Cry1_params[3]*cos(2*jtime_RC[i]*w-Cry1_params[5]-phi_i_add_RC[i]));
        f2 = freq_scaleNr1d1*(Nr1d1_params[1]/2+amp_scal*Nr1d1_params[2]*cos(jtime_RC[i]*w-Nr1d1_params[4]-phi_i_add_RC[i])+amp_scal*Nr1d1_params[3]*cos(2*jtime_RC[i]*w-Nr1d1_params[5]-phi_i_add_RC[i]));
        mu1[i] = b1*f1;
        mu2[i] = b2*f2;
        r1[i] = f1;
        r2[i] = f2;
    } 
    for ( i in 1:N2 ) {
        b3 = burstCry1*(AreaNormed_BC[i])^beta_v_Cry1;
        b4 = burstBmal1*(AreaNormed_BC[i])^beta_v_Bmal1;
        f3 = freq_scaleCry1*(Cry1_params[1]/2+amp_scal*Cry1_params[2]*cos(jtime_BC[i]*w-Cry1_params[4]-phi_i_add_BC[i])+amp_scal*Cry1_params[3]*cos(2*jtime_BC[i]*w-Cry1_params[5]-phi_i_add_BC[i]));
        f4 = freq_scaleBmal1*(Bmal1_params[1]/2+amp_scal*Bmal1_params[2]*cos(jtime_BC[i]*w-Bmal1_params[4]-phi_i_add_BC[i])+amp_scal*Bmal1_params[3]*cos(2*jtime_BC[i]*w-Bmal1_params[5]-phi_i_add_BC[i]));
        mu3[i] = b3*f3;
        mu4[i] = b4*f4;
        r3[i] = f3;
        r4[i] = f4;
    } 
    for (n in 1:N1) {
        phi_i_add_RC[n] ~ von_mises(0, kappa);
    }
    for (n in 1:N2) {
        phi_i_add_BC[n] ~ von_mises(0, kappa);
    }
    CountsCry1_RC ~ neg_binomial_2( mu1 , r1 );
    CountsNr1d1_RC ~ neg_binomial_2( mu2 , r2 );
    CountsCry1_BC ~ neg_binomial_2( mu3 , r3 );
    CountsBmal1_BC ~ neg_binomial_2( mu4 , r4 );
}
generated quantities{ 
    vector[N1] log_lik1;
    vector[N1] log_lik2;
    vector[N2] log_lik3;
    vector[N2] log_lik4;
    real mu1;
    real mu2;
    real mu3;
    real mu4;
    real r1;
    real r2;
    real r3;
    real r4;
    real b1;
    real b2;
    real b3;
    real b4;
    real f1;
    real f2;
    real f3;
    real f4;
    for ( i in 1:N1 ) {
        b1 = burstCry1*(AreaNormed_RC[i])^beta_v_Cry1;
        b2 = burstNr1d1*(AreaNormed_RC[i])^beta_v_Nr1d1;
        f1 = freq_scaleCry1*(Cry1_params[1]/2+amp_scal*Cry1_params[2]*cos(jtime_RC[i]*w-Cry1_params[4]-phi_i_add_RC[i])+amp_scal*Cry1_params[3]*cos(2*jtime_RC[i]*w-Cry1_params[5]-phi_i_add_RC[i]));
        f2 = freq_scaleNr1d1*(Nr1d1_params[1]/2+amp_scal*Nr1d1_params[2]*cos(jtime_RC[i]*w-Nr1d1_params[4]-phi_i_add_RC[i])+amp_scal*Nr1d1_params[3]*cos(2*jtime_RC[i]*w-Nr1d1_params[5]-phi_i_add_RC[i]));
        mu1 = b1*f1;
        mu2 = b2*f2;
        r1 = f1;
        r2 = f2;
        log_lik1[i] = neg_binomial_2_lpmf(CountsCry1_RC[i] | mu1 , r1);
        log_lik2[i] = neg_binomial_2_lpmf(CountsNr1d1_RC[i] | mu2 , r2);
    } 
    for ( i in 1:N2 ) {
        b3 = burstCry1*(AreaNormed_BC[i])^beta_v_Cry1;
        b4 = burstBmal1*(AreaNormed_BC[i])^beta_v_Bmal1;
        f3 = freq_scaleCry1*(Cry1_params[1]/2+amp_scal*Cry1_params[2]*cos(jtime_BC[i]*w-Cry1_params[4]-phi_i_add_BC[i])+amp_scal*Cry1_params[3]*cos(2*jtime_BC[i]*w-Cry1_params[5]-phi_i_add_BC[i]));
        f4 = freq_scaleBmal1*(Bmal1_params[1]/2+amp_scal*Bmal1_params[2]*cos(jtime_BC[i]*w-Bmal1_params[4]-phi_i_add_BC[i])+amp_scal*Bmal1_params[3]*cos(2*jtime_BC[i]*w-Bmal1_params[5]-phi_i_add_BC[i]));
        mu3 = b3*f3;
        mu4 = b4*f4;
        r3 = f3;
        r4 = f4;
        log_lik3[i] = neg_binomial_2_lpmf(CountsCry1_BC[i] | mu3 , r3);
        log_lik4[i] = neg_binomial_2_lpmf(CountsBmal1_BC[i] | mu4 , r4);
    } 
    

}        

"""

#%%

sm = pickle.load(open('model_2.pkl', 'rb'))
fit = pickle.load(open('fit_2.pkl', 'rb')) 

a = fit.extract(permuted=True)

# load parameters from model M2
freq_scaleCry1 = np.mean(a['freq_scaleCry1'],axis=0)
freq_scaleNr1d1 = np.mean(a['freq_scaleNr1d1'],axis=0)
freq_scaleBmal1 = np.mean(a['freq_scaleBmal1'],axis=0)
beta_v_Cry1 = np.mean(a['beta_v_Cry1'],axis=0)
beta_v_Nr1d1 = np.mean(a['beta_v_Nr1d1'],axis=0)
beta_v_Bmal1 = np.mean(a['beta_v_Bmal1'],axis=0)
burstCry1 = np.mean(a['burstCry1'],axis=0)
burstNr1d1 = np.mean(a['burstNr1d1'],axis=0)
burstBmal1 = np.mean(a['burstBmal1'],axis=0)

dat = {
    'N1' : len(CountsNr1d1_RC),
    'CountsNr1d1_RC' : CountsNr1d1_RC,
    'CountsCry1_RC' : CountsCry1_RC,   
    'jtime_RC' : jtime_RC, 
    'AreaNormed_RC' : AreaNormed_RC, 
    'N2' : len(CountsBmal1_BC),
    'CountsBmal1_BC' : CountsBmal1_BC,
    'CountsCry1_BC' : CountsCry1_BC,    
    'jtime_BC' : jtime_BC,
    'AreaNormed_BC' : AreaNormed_BC,
    'kappa' : 2,
    'beta_v_Cry1' : beta_v_Cry1,
    'beta_v_Nr1d1' : beta_v_Nr1d1,
    'beta_v_Bmal1' : beta_v_Bmal1,
    'freq_scaleCry1' : freq_scaleCry1,
    'freq_scaleNr1d1' : freq_scaleNr1d1, 
    'freq_scaleBmal1' : freq_scaleBmal1, 
    'burstCry1' : burstCry1,
    'burstNr1d1' : burstNr1d1,
    'burstBmal1' : burstBmal1,
    'N_f' : len(Nr1d1_params),
    'Nr1d1_params' : Nr1d1_params,
    'Cry1_params' : Cry1_params,
    'Bmal1_params' : Bmal1_params,
    'w' : w
}
#%%

sm = pystan.StanModel(model_code=model1)

fit = sm.sampling(data=dat, seed=194838, iter=4000, chains=8, control=dict(adapt_delta=0.95))

with open('model_3.pkl', 'wb') as f:
    pickle.dump(sm, f)
    
with open('fit_3.pkl', 'wb') as g:
    pickle.dump(fit, g)
    
#%%
# MODEL4 

model1 = """
data{
    int<lower=1> N1;
    int CountsNr1d1_RC[N1];
    int CountsCry1_RC[N1];    
    real jtime_RC[N1];
    real AreaNormed_RC[N1];
    int<lower=1> N2;
    int CountsBmal1_BC[N2];
    int CountsCry1_BC[N2];    
    real jtime_BC[N2];
    real AreaNormed_BC[N2];
    real<lower=0> beta_v_Cry1;
    real<lower=0> beta_v_Nr1d1;
    real<lower=0> beta_v_Bmal1;
    real<lower=0> freq_scaleCry1;
    real<lower=0> freq_scaleNr1d1; 
    real<lower=0> freq_scaleBmal1; 
    int <lower=1> N_f;
    real Nr1d1_params[N_f];
    real Cry1_params[N_f];
    real Bmal1_params[N_f];
    real w;
}
parameters{
    cholesky_factor_cov[2] L_RC;
    cholesky_factor_cov[2] L_BC;
    real burstCry1;
    real burstNr1d1;
    real burstBmal1;
    real eta_Cry1RC[N1];
    real eta_Nr1d1RC[N1];
    real eta_Cry1BC[N2];
    real eta_Bmal1BC[N2];
}
transformed parameters {
    matrix[2, 2] cov_vec_RC;
    vector[2] mu_vec_RC;
    matrix[2, 2] cov_vec_BC;
    vector[2] mu_vec_BC;  
    real<lower=0> stdevCry1;
    real<lower=0> stdevNr1d1; 
    real<lower=0> stdevBmal1; 
    real<lower=-1, upper=1> corrRC;
    real<lower=-1, upper=1> corrBC;
    cov_vec_RC = L_RC*L_RC';
    cov_vec_BC = L_BC*L_BC';
    mu_vec_RC[1] = burstCry1;
    mu_vec_RC[2] = burstNr1d1;
    stdevCry1 = (cov_vec_RC[1,1])^0.5;
    stdevNr1d1 = (cov_vec_RC[2,2])^0.5; 
    corrRC = cov_vec_RC[1,2]/((stdevCry1)*(stdevNr1d1));
    mu_vec_BC[1] = burstCry1;
    mu_vec_BC[2] = burstBmal1;
    stdevCry1 = (cov_vec_BC[1,1])^0.5;
    stdevBmal1 = (cov_vec_BC[2,2])^0.5; 
    corrBC = cov_vec_BC[1,2]/((stdevCry1)*(stdevBmal1));      
}
model{
    vector[N1] mu1;
    vector[N1] mu2;
    vector[N2] mu3;
    vector[N2] mu4;
    vector[N1] r1;
    vector[N1] r2;
    vector[N2] r3;
    vector[N2] r4;
    real b1;
    real b2;
    real b3;
    real b4;
    real f1;
    real f2;
    real f3;
    real f4;
    vector[2] prior_eta;
    vector[2] rescaled_eta;
    for ( i in 1:N1 ) {
        prior_eta[1] = eta_Cry1RC[i];
        prior_eta[2] = eta_Nr1d1RC[i];
        rescaled_eta = mu_vec_RC + L_RC*prior_eta;
        prior_eta ~ normal(0, 1);
        b1 = exp(beta_v_Cry1*log(AreaNormed_RC[i])+rescaled_eta[1]);
        b2 = exp(beta_v_Nr1d1*log(AreaNormed_RC[i])+rescaled_eta[2]);
        f1 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_RC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_RC[i]*w-Cry1_params[5]));
        f2 = freq_scaleNr1d1*(Nr1d1_params[1]/2+Nr1d1_params[2]*cos(jtime_RC[i]*w-Nr1d1_params[4])+Nr1d1_params[3]*cos(2*jtime_RC[i]*w-Nr1d1_params[5]));
        mu1[i] = b1*f1;
        mu2[i] = b2*f2;
        r1[i] = f1;
        r2[i] = f2;
    } 
    for ( i in 1:N2 ) {
        prior_eta[1] = eta_Cry1BC[i];
        prior_eta[2] = eta_Bmal1BC[i];
        rescaled_eta = mu_vec_BC + L_BC*prior_eta;
        prior_eta ~ normal(0, 1);
        b3 = exp(beta_v_Cry1*log(AreaNormed_BC[i])+rescaled_eta[1]);
        b4 = exp(beta_v_Bmal1*log(AreaNormed_BC[i])+rescaled_eta[2]);
        f3 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_BC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_BC[i]*w-Cry1_params[5]));
        f4 = freq_scaleBmal1*(Bmal1_params[1]/2+Bmal1_params[2]*cos(jtime_BC[i]*w-Bmal1_params[4])+Bmal1_params[3]*cos(2*jtime_BC[i]*w-Bmal1_params[5]));
        mu3[i] = b3*f3;
        mu4[i] = b4*f4;
        r3[i] = f3;
        r4[i] = f4;
    } 
    CountsCry1_RC ~ neg_binomial_2( mu1 , r1 );
    CountsNr1d1_RC ~ neg_binomial_2( mu2 , r2 );
    CountsCry1_BC ~ neg_binomial_2( mu3 , r3 );
    CountsBmal1_BC ~ neg_binomial_2( mu4 , r4 );
    burstCry1 ~ normal(0, 100);
    burstNr1d1 ~ normal(0, 100);
    burstBmal1 ~ normal(0, 100);
    L_RC ~ lkj_corr_cholesky(4.0);
    L_BC ~ lkj_corr_cholesky(4.0);
}
generated quantities{ 
    vector[N1] log_lik1;
    vector[N1] log_lik2;
    vector[N2] log_lik3;
    vector[N2] log_lik4;
    real mu1;
    real mu2;
    real mu3;
    real mu4;
    real r1;
    real r2;
    real r3;
    real r4;
    real b1;
    real b2;
    real b3;
    real b4;
    real f1;
    real f2;
    real f3;
    real f4;
    vector[2] prior_eta;
    vector[2] rescaled_eta;
    for ( i in 1:N1 ) {
 prior_eta[1] = eta_Cry1RC[i];
        prior_eta[2] = eta_Nr1d1RC[i];
        rescaled_eta = mu_vec_RC + L_RC*prior_eta;
        b1 = exp(beta_v_Cry1*log(AreaNormed_RC[i])+rescaled_eta[1]);
        b2 = exp(beta_v_Nr1d1*log(AreaNormed_RC[i])+rescaled_eta[2]);
        f1 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_RC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_RC[i]*w-Cry1_params[5]));
        f2 = freq_scaleNr1d1*(Nr1d1_params[1]/2+Nr1d1_params[2]*cos(jtime_RC[i]*w-Nr1d1_params[4])+Nr1d1_params[3]*cos(2*jtime_RC[i]*w-Nr1d1_params[5]));
        mu1 = b1*f1;
        mu2 = b2*f2;
        r1 = f1;
        r2 = f2;
        log_lik1[i] = neg_binomial_2_lpmf(CountsCry1_RC[i] | mu1 , r1);
        log_lik2[i] = neg_binomial_2_lpmf(CountsNr1d1_RC[i] | mu2 , r2);
    } 
    for ( i in 1:N2 ) {
       prior_eta[1] = eta_Cry1BC[i];
        prior_eta[2] = eta_Bmal1BC[i];
        rescaled_eta = mu_vec_BC + L_BC*prior_eta;
        b3 = exp(beta_v_Cry1*log(AreaNormed_BC[i])+rescaled_eta[1]);
        b4 = exp(beta_v_Bmal1*log(AreaNormed_BC[i])+rescaled_eta[2]);
        f3 = freq_scaleCry1*(Cry1_params[1]/2+Cry1_params[2]*cos(jtime_BC[i]*w-Cry1_params[4])+Cry1_params[3]*cos(2*jtime_BC[i]*w-Cry1_params[5]));
        f4 = freq_scaleBmal1*(Bmal1_params[1]/2+Bmal1_params[2]*cos(jtime_BC[i]*w-Bmal1_params[4])+Bmal1_params[3]*cos(2*jtime_BC[i]*w-Bmal1_params[5]));
        mu3 = b3*f3;
        mu4 = b4*f4;
        r3 = f3;
        r4 = f4;
        log_lik3[i] = neg_binomial_2_lpmf(CountsCry1_BC[i] | mu3 , r3);
        log_lik4[i] = neg_binomial_2_lpmf(CountsBmal1_BC[i] | mu4 , r4);
    } 
    

}        
"""
#%%
# load parameters from model M2

sm = pickle.load(open('model_2.pkl', 'rb'))
fit = pickle.load(open('fit_2.pkl', 'rb')) 

a = fit.extract(permuted=True)    

freq_scaleCry1 = np.mean(a['freq_scaleCry1'],axis=0)
freq_scaleNr1d1 = np.mean(a['freq_scaleNr1d1'],axis=0)
freq_scaleBmal1 = np.mean(a['freq_scaleBmal1'],axis=0)
beta_v_Cry1 = np.mean(a['beta_v_Cry1'],axis=0)
beta_v_Nr1d1 = np.mean(a['beta_v_Nr1d1'],axis=0)
beta_v_Bmal1 = np.mean(a['beta_v_Bmal1'],axis=0)


dat = {
    'N1' : len(CountsNr1d1_RC),
    'CountsNr1d1_RC' : CountsNr1d1_RC,
    'CountsCry1_RC' : CountsCry1_RC,   
    'jtime_RC' : jtime_RC, 
    'AreaNormed_RC' : AreaNormed_RC, 
    'N2' : len(CountsBmal1_BC),
    'CountsBmal1_BC' : CountsBmal1_BC,
    'CountsCry1_BC' : CountsCry1_BC,    
    'jtime_BC' : jtime_BC,
    'AreaNormed_BC' : AreaNormed_BC,
    'beta_v_Cry1' : beta_v_Cry1,
    'beta_v_Nr1d1' : beta_v_Nr1d1,
    'beta_v_Bmal1' : beta_v_Bmal1,
    'freq_scaleCry1' : freq_scaleCry1,
    'freq_scaleNr1d1' : freq_scaleNr1d1, 
    'freq_scaleBmal1' : freq_scaleBmal1, 
    'N_f' : len(Nr1d1_params),
    'Nr1d1_params' : Nr1d1_params,
    'Cry1_params' : Cry1_params,
    'Bmal1_params' : Bmal1_params,
    'w' : w
}

sm = pystan.StanModel(model_code=model1)

fit = sm.sampling(data=dat, seed=194838, iter=2000, chains=4, control=dict(adapt_delta=0.95))

with open('model_4.pkl', 'wb') as f:
    pickle.dump(sm, f)
    
with open('fit_4.pkl', 'wb') as g:
    pickle.dump(fit, g)
    