#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from scipy import stats
from scipy.stats.stats import pearsonr

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

# define colours for scatter/contour plots

alpha = 0.3

posits = (np.linspace(0,255,7)).astype('int64')
colors_conts = np.zeros((7,4))

for i in range(len(colors_conts)):
    colors_conts[i,:] = cm.brg(posits[i])
    colors_conts[i,3] = alpha  
    
# define upper x and y limits of plots
    
max_xRC = 125
max_yRC = 50

max_xBC = 125
max_yBC = 50

dotsize = 0.5

# calculate KDE estimate of bivariate mRNA distribution

def get_count_kde(counts_x,counts_y,jtime,Timepoint_vec):
    
    x1_vec = range(100)
    x2_vec = range(100)
    kde_tot_dat = np.zeros((len(x1_vec),len(x2_vec),len(Timepoint_vec)))
    
    for z in range(len(Timepoint_vec)):
        x = counts_x[jtime==Timepoint_vec[z]].reshape(1,-1)
        y = counts_y[jtime==Timepoint_vec[z]].reshape(1,-1)
        temp = np.concatenate((x,y),axis=0)
        kde = stats.gaussian_kde(np.concatenate((x,y),axis=0))
        for i,x1_curr in enumerate(x1_vec):
            for j,x2_curr in enumerate(x2_vec):
                 kde_tot_dat[j,i,z] = kde((x1_curr,x2_curr))  
    
    return kde_tot_dat

# define plotting function
    
def plt_2D(x,y,kde_tot_dat,max_x,max_y,Timevec,i,i_col):
    plt.scatter(x, y,s=dotsize,alpha=alpha,color=colors_conts[i_col])
    plt.contour((kde_tot_dat[:,:,i]),colors = 'k',levels = 10,alpha=0.4,linewidths=0.5) 
    plt.xlim((0, max_x))
    plt.ylim((0, max_y))
    rval, pval = pearsonr(x,y)
    plt.title('%(timepoint)i h. R = %(correlation)1.2f' % {"timepoint":Timevec[i],"correlation": rval})
    
    
kde_tot_dat = get_count_kde(CountsBmal1_BC,CountsCry1_BC,jtime_BC,TimevecBC)   

fig = plt.figure(figsize=(10/2.54,25/2.54))

for i in range(0,7):
    plt.subplot(7,2,2*i+1)
    x = CountsBmal1_BC[jtime_BC==TimevecBC[i]] 
    y = CountsCry1_BC[jtime_BC==TimevecBC[i]]
    plt_2D(x,y,kde_tot_dat,max_xBC,max_yBC,TimevecBC,i,i)
    plt.xticks(np.arange(0, max_xBC+1, step=25))
    plt.yticks(np.arange(0, max_yBC+1, step=10))
    if i!=6:
        plt.gca().axes.xaxis.set_ticklabels([]) 
    if i == 3:
        plt.ylabel(r'Cry1 mRNA/cell')
    if i == 6:
        plt.xlabel(r'Bmal1 mRNA/cell')
    
kde_tot_dat = get_count_kde(CountsNr1d1_RC,CountsCry1_RC,jtime_RC,TimevecRC)

for i in range(0,6):
    plt.subplot(7,2,2*i+4)
    x = CountsNr1d1_RC[jtime_RC==TimevecRC[i]] 
    y = CountsCry1_RC[jtime_RC==TimevecRC[i]] 
    plt_2D(x,y,kde_tot_dat,max_xBC,max_yBC,TimevecRC,i,i+1)
    plt.xticks(np.arange(0, max_xBC+1, step=25))
    plt.yticks(np.arange(0, max_yBC+1, step=10))
    if i!=5:
        plt.gca().axes.xaxis.set_ticklabels([]) 
        plt.gca().axes.yaxis.set_ticklabels([])
    if i == 5:
        plt.xlabel('Nr1d1 mRNA/cell')
    
#%%
# plot area vs mRNA count and regression for each gene

def get_binned_and_regression(area,counts):

    lims = np.percentile(area, [5, 95])
               
    bin_nums = 20
    vol_bins = np.linspace(np.log(lims[0]),np.log(lims[1]),bin_nums)
    cond_mean = np.zeros(bin_nums-1)
    
    for i in range(bin_nums-1):
        filt = np.logical_and(np.log(area)<vol_bins[i+1],np.log(area)>vol_bins[i])
        counts_curr = counts[filt]
        cond_mean[i] = np.log(np.mean(counts_curr))
    
    x = vol_bins[:-1]+0.5*(vol_bins[1]-vol_bins[0])
    y = cond_mean
    
    xfilt = x[~np.isnan(y)]
    yfilt = y[~np.isnan(y)]
    xfilt = xfilt[~np.isinf(yfilt)]
    yfilt = yfilt[~np.isinf(yfilt)]
        
    c1, c2 = np.polyfit(xfilt,yfilt,1)
    
    return x,y,c1,c2

plt.style.use('seaborn-ticks')
fig = plt.figure(figsize=(6/2.54,6/2.54))

x_binned,y_binned,c1,c2 = get_binned_and_regression(AreaNormed_RC,CountsNr1d1_RC)
plt.scatter(x_binned,y_binned,color='blue',s=2)
plt.plot(x_binned,c2+x_binned*c1,color='blue',label = 'Nr1d1. y = %(c1)1.2f*x + %(c2)1.2f' % {"c1":c1,"c2": c2})

x_binned,y_binned,c1,c2 = get_binned_and_regression(AreaCry1TOT,CountsCry1TOT)
plt.scatter(x_binned,y_binned,color='green',s=2)
plt.plot(x_binned,c2+x_binned*c1,color='green',label = 'Cry1. y = %(c1)1.2f*x + %(c2)1.2f' % {"c1":c1,"c2": c2})

x_binned,y_binned,c1,c2 = get_binned_and_regression(AreaNormed_BC,CountsBmal1_BC)
plt.scatter(x_binned,y_binned,color='red',s=2)
plt.plot(x_binned,c2+x_binned*c1,color='red',label = 'Bmal1. y = %(c1)1.2f*x + %(c2)1.2f' % {"c1":c1,"c2": c2})

plt.ylim([1.8,4.5])
legend = plt.legend(loc='upper left', shadow=True)
plt.xlabel(r'log(Normalised Area)')
plt.ylabel('log(E[m|A])')

plt.tight_layout()  

