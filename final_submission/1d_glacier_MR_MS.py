#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import save
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import GlacierSim
from GlacierSim import glacier_simulation,closest,resp_time,power_law
import pandas as pd
from scipy.optimize import curve_fit

# This is the main function file
# All functions not part of the plotting can be found in GlacierSim.py, including the model (the glacier_simulation function)
# Other files are included as data files, since we had to run the model twice - one for "up" and one for "down", and then we saved the 
    #data and reimported in after the second run. A quick description of included files:
    # - down_data.npy : the results of running the "running the model" section, which takes too long to run each time we wanted to edit the code for something else.
    # - final_rt_down.npy/final_rt_up.npy : numpy arrays of the response times for the up/down runs to allow for creation of figure 1
    # - MATPLOTLIB_RCPARAMS.sty : file to make the plots all have the same latex-looking style
    # - powers_up.csv : csv file with the best fit parameters for the "up" runs
    # - responsetimes.csv : response times for the bmax runs. See that section for extra notes.
    # - rs.npy: response times from the "up" data, saved as a np array


#--------running the model------ 
#this runs fine, it just takes a very long time, so we've included the file with the important data as well (down_data.npy).

#totL  = 30000 # total length of the domain [m]
#dx    =   200 # grid size [m]
#ntpy  =   300 # number of timesteps per year
#elalist = np.array([1550., 1600., 1650., 1700.,1750., 1800.])  # m -> default 1800., 1750., 1900., 1800.,
#elayear = np.array([1000., 1000., 1000., 1000.,1000., 1000.])  # years        
#dbdh    =    0.005    # [m/m]/yr default 0.007
#maxb    =    2.      # m/yr default 2

#simdict= glacier_simulation(totL,dx,ntpy,elalist,elayear,dbdh,maxb)

#Running model for every value of dbdh, return response times in a dictionary
#r_times_down={}
#for d_bh in [0.005,0.006,0.0075,0.01,0.015,0.02,0.025,0.03]:
#    simdict = glacier_simulation(totL,dx,ntpy,elalist,elayear,d_bh,maxb)
#    resp_t  = resp_time(simdict["mass"],len(elalist),elayear[0],ntpy,"down")
#    r_times_down[str(d_bh)]=list(resp_t)
    
#Response times given by dbdh for a series of ELA levels,  rearrange to give by ELA levels for a series of dbdh's
#r_times_down_2=r_times_down.copy()
#r_times_down_1={}
betas=[0.005,0.006,0.0075,0.01,0.015,0.02,0.025,0.03]
#rts=[]
#for d_bh2 in betas:
#    rts=[]
#    for i in np.arange(0,6):
#        rts=np.append(rts,r_times_down_2[str(d_bh2)][i])
#        r_times_down_1[str(d_bh2)]=rts

#This is repeated for both "up" - inc. mass/dec. ela and "down" dec.mass/inc ela runs. Currently set up for a 'down' run, need to import "up" data
#np.save('down_data.npy',r_times_down_1)

r_times_down_1=np.load('down_data.npy',allow_pickle=True).item() #when running the full model, comment out this line

r_time_up=np.load('rs.npy',allow_pickle=True) #response times from the "up" data, saved as a np array
params_up=pd.read_csv("powers_up.csv") #Parameters for the best fit lines for the "up" data

df_d=pd.DataFrame(data=r_times_down_1)
df_d.insert(0,"ind",value=[0,1,2,3,4,5])

#Different ELA lines (up)
r1_u=r_time_up[:8]
r2_u=r_time_up[8:16]
r3_u=r_time_up[16:24]
r4_u=r_time_up[24:32]
r5_u=r_time_up[32:40]

#Different ELA lines (down)
r1=(df_d[df_d['ind']==0].to_numpy()).ravel()
r2=(df_d[df_d['ind']==1].to_numpy()).ravel()
r3=(df_d[df_d['ind']==2].to_numpy()).ravel()
r4=(df_d[df_d['ind']==3].to_numpy()).ravel()
r5=(df_d[df_d['ind']==4].to_numpy()).ravel()

#different best fit parameters (down)
popt_1, pcov_1 = curve_fit(power_law, betas, r1[1:],bounds=([0,0], [50, 30]))
popt_2, pcov_2 = curve_fit(power_law, betas, r2[1:],bounds=([0,0], [50, 30]))
popt_3, pcov_3 = curve_fit(power_law, betas, r3[1:],bounds=([0,0], [50, 30]))
popt_4, pcov_4 = curve_fit(power_law, betas, r4[1:],bounds=([0,0], [50, 30]))
popt_5, pcov_5 = curve_fit(power_law, betas, r5[1:],bounds=([0,0], [50, 30]))

#labels + style settings
labellist1=[r'$1850m \rightarrow 1800m$',r'$1800m \rightarrow 1750m$',r'$1750m \rightarrow 1700m$',r'$1700m \rightarrow 1650m$',r'$1650m \rightarrow 1600m$']
labellist2=[r'$1800m \rightarrow 1850m$',r'$1750m \rightarrow 1800m$',r'$1700m \rightarrow 1750m$',r'$1650m \rightarrow 1700m$',r'$1600m \rightarrow 1650m$']

plt.style.use("MATPLOTLIB_RCPARAMS.sty")
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
fig,axs=plt.subplots(ncols=2,nrows=1,figsize=(16,8))
color='skyblue'

#Plotting down response times
axs[1].plot(betas,r1[1:],color,label='Response times')
axs[1].plot(betas,r2[1:],color)
axs[1].plot(betas,r3[1:],color)
axs[1].plot(betas,r4[1:],color)
axs[1].plot(betas,r5[1:],color)

#Plotting down best fit lines
axs[1].plot(np.array(betas), power_law(betas, *popt_1),"tab:blue"  ,label=labellist2[0])#1550-1600
axs[1].plot(np.array(betas), power_law(betas, *popt_2),"tab:orange",label=labellist2[1])#1600-1650
axs[1].plot(np.array(betas), power_law(betas, *popt_3),"tab:green" ,label=labellist2[2])#1650-1700
axs[1].plot(np.array(betas), power_law(betas, *popt_4),"tab:red"   ,label=labellist2[3])#1700-1750
axs[1].plot(np.array(betas), power_law(betas, *popt_5),"tab:purple",label=labellist2[4])#1750-1800

axs[1].set_yscale("log")
axs[1].set_xscale("log")
axs[1].legend()
axs[1].set_xlabel(r'$\beta~[yr^{-1}]$')
axs[1].set_ylabel(r'$\tau~[yr]$')
axs[1].set_title('b.) Response time: increasing ELA/shrinking glacier',fontsize=15)

#plotting up response times
axs[0].plot(betas,r1_u,color,label='Response times')
axs[0].plot(betas,r2_u,color)
axs[0].plot(betas,r3_u,color)
axs[0].plot(betas,r4_u,color)
axs[0].plot(betas,r5_u,color)

#plotting up best fit lines
axs[0].plot(np.array(betas), power_law(betas, *np.array(params_up.iloc[:,1])),label=labellist1[0])
axs[0].plot(np.array(betas), power_law(betas, *np.array(params_up.iloc[:,2])),label=labellist1[1])
axs[0].plot(np.array(betas), power_law(betas, *np.array(params_up.iloc[:,3])),label=labellist1[2])
axs[0].plot(np.array(betas), power_law(betas, *np.array(params_up.iloc[:,4])),label=labellist1[3])
axs[0].plot(np.array(betas), power_law(betas, *np.array(params_up.iloc[:,5])),label=labellist1[4])

axs[0].set_yscale("log")
axs[0].set_xscale("log")
axs[0].legend()
axs[0].set_xlabel(r'$\beta~[yr^{-1}]$')
axs[0].set_ylabel(r'$\tau~[yr]$')
axs[0].set_title('a.) Response time: decreasing ELA/growing glacier',fontsize=15)

fig.tight_layout()
plt.savefig("power_plot.pdf") #Figure 2 in the report

#Bmax plots were created on an older script, one without the option to run it in a loop and save the results each time, 
# so I had to manually copy the response time values into an excel file, and then upload it here as a csv.
resp_t_1=pd.read_csv("responsetimes.csv")

#pulling the values needed to plot
smb=resp_t_1["Unnamed: 0"][:7]
resp_t_1.insert(0,"indix",np.arange(0,14))
dec_ela=resp_t_1[:][:7]
inc_ela=resp_t_1[:][7:]

#into a dictionary for easier removal
rt_m_up={}
rt_m_down={}
for i in range(7):
    rt_m_up[str(resp_t_1['Unnamed: 0'][i])]=np.array(resp_t_1.iloc[i,2:])
    rt_m_down[str(resp_t_1['Unnamed: 0'][i])]=np.array(resp_t_1.iloc[i+7,2:])

f, axs1 = plt.subplots(ncols=2, nrows=2,figsize=(15,10))

#f.suptitle('Response times for growing and shrinking glaciers',fontsize=20)
plt.style.use("MATPLOTLIB_RCPARAMS.sty")
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

rt_up=rt_m_up
rt_down=rt_m_down
rt_up_beta=np.load("final_rt_up.npy",allow_pickle=True).item()
rt_down_beta=np.load("final_rt_down.npy",allow_pickle=True).item()

labellist1=[r'$1850m \rightarrow 1800m$',r'$1800m \rightarrow 1750m$',r'$1750m \rightarrow 1700m$',r'$1700m \rightarrow 1650m$',r'$1650m \rightarrow 1600m$']
labellist2=[r'$1800m \rightarrow 1850m$',r'$1750m \rightarrow 1800m$',r'$1700m \rightarrow 1750m$',r'$1650m \rightarrow 1700m$',r'$1600m \rightarrow 1650m$']

x=[float(e) for e in rt_up.keys()]
y=[float(e) for e in rt_up_beta.keys()]

for i in range(len(rt_up['0.25'])):
    axs1[1,0].plot(x,[rt_up[e][i] for e in rt_up.keys()],label=labellist1[i])
    axs1[0,0].plot(y,[rt_up_beta[e][i] for e in rt_up_beta.keys()],label=labellist1[i])

axs1[0,0].set_xlabel(r'$\beta~[yr^{-1}]$')
axs1[0,0].set_ylabel(r'$\tau~[yr]$')
axs1[0,0].set_title('a.) Response time: decreasing ELA/growing glacier',fontsize=15)
axs1[0,0].set_ylim([25,375])
axs1[0,0].legend()

axs1[1,0].set_xlabel(r'$b_{max}~[yr^{-1}]$')
axs1[1,0].set_ylabel(r'$\tau~[yr]$')
axs1[1,0].set_title('c.) Response time: decreasing ELA/growing glacier',fontsize=15)
axs1[1,0].set_ylim([25,275])
axs1[1,0].legend()

for i in range(len(rt_up['0.25'])):
    axs1[1,1].plot(x,[rt_down[e][i] for e in rt_up.keys()],label=labellist2[i])
    axs1[0,1].plot(y,[rt_down_beta[e][i] for e in rt_up_beta.keys()],label=labellist2[i])


axs1[0,1].set_title('b.) Response time: increasing ELA/shrinking glacier',fontsize=15)
axs1[0,1].set_xlabel(r'$\beta~[yr^{-1}]$')
axs1[0,1].set_ylabel(r'$\tau~[yr]$')
axs1[0,1].set_ylim([25,375])
axs1[0,1].legend()

axs1[1,1].set_title('d.) Response time: increasing ELA/shrinking glacier',fontsize=15)
axs1[1,1].set_xlabel(r'$b_{max}~[yr^{-1}]$')
axs1[1,1].set_ylabel(r'$\tau~[yr]$')
axs1[1,1].set_ylim([25,275])
axs1[1,1].legend()
f.tight_layout()

plt.savefig('RT_3.pdf') #Figure 1 in the report
