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
from psis import psisloo as ps

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
plt.style.use('seaborn-ticks')

#%%
# load model fits and calculate PSIS-LOO

sm = pickle.load(open('model_1.pkl', 'rb'))
fit = pickle.load(open('fit_1.pkl', 'rb'))     
a = fit.extract(permuted=True)
log_lik = np.hstack((a['log_lik1'],a['log_lik2'],a['log_lik3'],a['log_lik4']))
looM1, loosM1, ksM1 = ps(log_lik)

sm = pickle.load(open('model_2.pkl', 'rb'))
fit = pickle.load(open('fit_2.pkl', 'rb')) 
a = fit.extract(permuted=True)
log_lik = np.hstack((a['log_lik1'],a['log_lik2'],a['log_lik3'],a['log_lik4']))
looM2, loosM2, ksM2 = ps(log_lik)

sm = pickle.load(open('model_3.pkl', 'rb'))
fit = pickle.load(open('fit_3.pkl', 'rb')) 
a = fit.extract(permuted=True)
log_lik = np.hstack((a['log_lik1'][1500:2000],a['log_lik2'][1500:2000],a['log_lik3'][1500:2000],a['log_lik4'][1500:2000]))
looM3, loosM3, ksM3 = ps(log_lik)

sm = pickle.load(open('model_4.pkl', 'rb'))
fit = pickle.load(open('fit_4.pkl', 'rb')) 
a = fit.extract(permuted=True)
log_lik = np.hstack((a['log_lik1'],a['log_lik2'],a['log_lik3'],a['log_lik4']))
looM4, loosM4, ksM4 = ps(log_lik)

fig = plt.figure(figsize=(5/2.54,3.5/2.54))
plt.bar(range(4),[looM1,looM2,looM3,looM4],tick_label = ['M1','M2','M3','M4'])
plt.ylim([-44250,-42250])
plt.ylabel('PSIS-LOO')
plt.tight_layout()   

#%%
# load in required parameters from model M2 and model M4

sm = pickle.load(open('model_2.pkl', 'rb'))
fit = pickle.load(open('fit_2.pkl', 'rb')) 
    
a = fit.extract(permuted=True)

freq_scaleCry1 = a['freq_scaleCry1']
freq_scaleNr1d1 = a['freq_scaleNr1d1']
freq_scaleBmal1 = a['freq_scaleBmal1']
beta_Cry1 = a['beta_v_Cry1']
beta_Nr1d1 = a['beta_v_Nr1d1']
beta_Bmal1 = a['beta_v_Bmal1']

sm = pickle.load(open('model_4.pkl', 'rb'))
fit = pickle.load(open('fit_4.pkl', 'rb')) 
    
a = fit.extract(permuted=True)

burstalphaCry1 = a['burstCry1']
burstalphaNr1d1 = a['burstNr1d1']
burstalphaBmal1 = a['burstBmal1']
stdevCry1 = a['stdevCry1']
stdevNr1d1 = a['stdevNr1d1']
stdevBmal1 = a['stdevBmal1']
corrRC = a['corrRC']
corrBC = a['corrBC']

Nr1d1_params, Cry1_params, Bmal1_params = pickle.load(open('FourierParams.pkl', 'rb'))
#%%
# plot posterior parameter values

fig = plt.figure(figsize=(17.8/2.54,15/2.54))

xtest = np.linspace(17,42,100)

def func(x, A0, A1, A2, phi_1, phi_2):
    w = 2 * np.pi / 24 
    y = A0/2 + A1*np.cos(w*x-phi_1) + A2*np.cos(2*w*x-phi_2)
    return y.flatten()

def plot_burstfreq(xtest,Fourier_params,freq_scale):
    pred_lower = func(xtest, *Fourier_params)*np.percentile(freq_scale, 5)
    pred_median = func(xtest, *Fourier_params)*np.percentile(freq_scale, 50)
    pred_upper = func(xtest, *Fourier_params)*np.percentile(freq_scale, 95)
    return pred_lower, pred_median, pred_upper

plt.subplot(5,4,5)
pred_lower, pred_median, pred_upper = plot_burstfreq(xtest,Nr1d1_params,freq_scaleNr1d1)
plt.plot(xtest, pred_median,color='b',label='Nr1d1')
plt.fill_between(xtest, pred_lower, pred_upper,alpha=0.5)

pred_lower, pred_median, pred_upper = plot_burstfreq(xtest,Cry1_params,freq_scaleCry1)
plt.plot(xtest, pred_median,color='g',label='Cry1')
plt.fill_between(xtest, pred_lower, pred_upper,alpha=0.5)

pred_lower, pred_median, pred_upper = plot_burstfreq(xtest,Bmal1_params,freq_scaleBmal1)
plt.plot(xtest, pred_median,color='r',label='Bmal1')
plt.fill_between(xtest, pred_lower, pred_upper,alpha=0.5)

plt.title(r'Burst freq (mRNA lifespan$^{-1}$)')
plt.xlabel('Time(hours)')


def plot_posterior(Nr1d1,Cry1,Bmal1,x_lower,x_upper,y_upper,i):
    
    plt.subplot(5,4,i)
    
    xx = np.linspace(x_lower, x_upper, 1000) 
    
    kde = stats.gaussian_kde(Nr1d1)
    plt.plot(xx, kde(xx)/sum(kde(xx)),color='b',label='Nr1d1')
    
    kde = stats.gaussian_kde(Cry1)
    plt.plot(xx, kde(xx)/sum(kde(xx)),color='g',label='Cry1')  

    kde = stats.gaussian_kde(Bmal1)
    plt.plot(xx, kde(xx)/sum(kde(xx)),color='r',label='Bmal1')
    
    plt.xlim((x_lower,x_upper))
    plt.ylim((0,y_upper))
    plt.yticks([])

def mean_burstsize(mu,std):

    avg = np.exp(mu+std**2/2)

    return avg    

plot_posterior(mean_burstsize(burstalphaNr1d1,stdevNr1d1),mean_burstsize(burstalphaCry1,stdevCry1),mean_burstsize(burstalphaBmal1,stdevBmal1),0,15,0.25,6)
plt.title(r'Avg burst size')
plt.ylabel('Density')
legend = plt.legend(loc='upper right', shadow=True)

plot_posterior(beta_Nr1d1,beta_Cry1,beta_Bmal1,0.4,0.8,0.015,7)
plt.title(r'$\beta$')

plt.subplot(5,4,8)  

lower = -1
upper = 1

xx = np.linspace(lower, upper, 1000) 

kde = stats.gaussian_kde(corrRC)
plt.plot(xx, kde(xx)/sum(kde(xx)),color='m',label='RC')
kde = stats.gaussian_kde(corrBC)
plt.plot(xx, kde(xx)/sum(kde(xx)),color='c',label='BC')

legend = plt.legend(loc='upper right', shadow=True)

plt.ylim((0,0.03))
plt.yticks([])

plt.xlim((lower,upper))
plt.xticks(np.linspace(lower, upper, 5))

plt.title(r'$\rho$')

#plt.tight_layout()
#%%
# load smFISH data

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

# load in required parameters from model M2 and model M4
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

mu_RC = np.mean(a['mu_vec_RC'],axis=0)#.reshape(-1,1)
Cov_RC = np.mean(a['cov_vec_RC'],axis=0)

mu_BC = np.mean(a['mu_vec_BC'],axis=0)#.reshape(-1,1)
Cov_BC = np.mean(a['cov_vec_BC'],axis=0)

Nr1d1_params, Cry1_params, Bmal1_params = pickle.load(open('FourierParams.pkl', 'rb'))
#%%

def kde_sims(params_x,params_y,freq_scale_x,freq_scale_y,beta_x,beta_y,mu,Cov,area,time_vec):
    
    time_vec = np.array([21,33])

    x1_vec = range(100)
    x2_vec = range(100)

    its = 15
    
    kde_sims = np.zeros((len(x1_vec),len(x2_vec),len(time_vec)))
    
    y1 = np.zeros(its*len(area))
    y2 = np.zeros(its*len(area))
    
    area_sims = np.repeat(area, its, axis=None)

    for k, time_curr in enumerate(time_vec):
        for n in range(len(area_sims)):
            eta = np.random.multivariate_normal(mu,Cov)
            r1 = freq_scale_x*(params_x[0]/2+params_x[1]*np.cos(time_curr*w-params_x[3])+params_x[2]*np.cos(2*time_curr*w-params_x[4]));
            r2 = freq_scale_y*(params_y[0]/2+params_y[1]*np.cos(time_curr*w-params_y[3])+params_y[2]*np.cos(2*time_curr*w-params_y[4]));
            b1 = np.exp(beta_x*np.log(area_sims[n])+eta[0]);
            b2 = np.exp(beta_y*np.log(area_sims[n])+eta[1]);
            m1 = b1*r1
            m2 = b2*r2
            p1 = r1/(m1+r1)
            p2 = r2/(m2+r2)
            y1[n] = np.random.negative_binomial(r1, p1)
            y2[n] = np.random.negative_binomial(r2, p2)
        
        kde = stats.gaussian_kde(np.vstack((y1,y2)),bw_method='silverman')
        for i, x1_curr in enumerate(x1_vec):
            print(i)
            for j, x2_curr in enumerate(x2_vec):
                val = np.array([[x1_curr],[x2_curr]])
                kde_sims[j,i,k] = kde(val)
    return kde_sims            

time_vec = np.array([21,33])
kde_sims_BC = kde_sims(Cry1_params,Bmal1_params,freq_scaleCry1,freq_scaleBmal1,beta_Cry1,beta_Bmal1,mu_BC,Cov_BC,AreaNormed_BC,time_vec)
kde_sims_RC = kde_sims(Cry1_params,Nr1d1_params,freq_scaleCry1,freq_scaleNr1d1,beta_Cry1,beta_Nr1d1,mu_RC,Cov_RC,AreaNormed_RC,time_vec)
         
#%%
def kde_empirical(Counts_x,Counts_y,jtime,time_vec):
    
    x1_vec = range(100)
    x2_vec = range(100)
    
    kde_dat = np.zeros((len(x1_vec),len(x2_vec),len(time_vec)))

    for k in range(len(time_vec)):
        x = (Counts_x[jtime==time_vec[k]]).reshape(1,-1)
        y = (Counts_y[jtime==time_vec[k]]).reshape(1,-1)
        kde = stats.gaussian_kde(np.concatenate((x,y),axis=0))
        for i, x1_curr in enumerate(x1_vec):
            print(i)
            for j, x2_curr in enumerate(x2_vec):
                val = np.array([[x1_curr],[x2_curr]])
                kde_dat[j,i,k] = kde(val)
    return kde_dat

kde_dat_BC = kde_empirical(CountsBmal1_BC,CountsCry1_BC,jtime_BC,time_vec) 
kde_dat_RC = kde_empirical(CountsNr1d1_RC,CountsCry1_RC,jtime_RC,time_vec)               
                
#%%
#fig = plt.figure(figsize=(3*len(time_vec),3))
for z in range(len(time_vec)):
    plt.subplot(5,4,4*z+13)   
    plt.contour((kde_dat_BC[:,:,z]),cmap = 'jet',levels = 10)        
    plt.ylabel('Cry1 mRNA')
    if z == 0:
        plt.title('Data') 
    if z != 1:
        plt.gca().axes.xaxis.set_ticklabels([])
    if z ==1:
        plt.xlabel('Bmal1 mRNA') 
    plt.ylim([0, 40])
    plt.xticks(np.arange(0, 101, step=25))

for z in range(len(time_vec)):
    plt.subplot(5,4,4*z+14)   
    plt.contour(np.transpose(kde_sims_BC[:,:,z]),cmap = 'jet',levels = 10)        
    if z == 0:
        plt.title('Model') 
    if z != 1:
        plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    if z == 1:
        plt.xlabel('Bmal1 mRNA') 
    plt.ylim([0, 40])
    plt.xticks(np.arange(0, 101, step=25))       
    
    
for z in range(len(time_vec)):
    plt.subplot(5,4,4*z+15)   
    plt.contour((kde_dat_RC[:,:,z]),cmap = 'jet',levels = 10)        
    plt.ylabel('Cry1 mRNA')
    if z == 0:
        plt.title('Data') 
    if z != 1:
        plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    if z == 1:
        plt.xlabel('Nr1d1 mRNA') 
    plt.ylim([0, 40])
    plt.xticks(np.arange(0, 101, step=25))

for z in range(len(time_vec)):
    plt.subplot(5,4,4*z+16)   
    plt.contour(np.transpose(kde_sims_RC[:,:,z]),cmap = 'jet',levels = 10)        
    if z == 0:
        plt.title('Model') 
    if z != 1:
        plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    if z ==1:
        plt.xlabel('Nr1d1 mRNA') 
    plt.ylim([0, 40])
    plt.xticks(np.arange(0, 101, step=25))    
    
    
plt.tight_layout()    