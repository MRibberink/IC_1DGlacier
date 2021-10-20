import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import save

ZeroFluxBoundary = True # either no-flux (True) or No-6ary (False)
FluxAtPoints     = False
StopWhenOutOfDomain = True                                           
ndyfigure = 5           # number of years between a figure frame                        
rho   =    917.      # kg/m3
g     =    9.80665 # m/s2
fd    =    1.9E-24 # # pa-3 s-1 # this value and dimension is only correct for n=3
fs    =    5.7E-20 # # pa-3 m2 s-1 # this value and dimension is only correct for n=3
cd    = 2/5*(rho*g)**3*fd  # <<< this must be adjusted according to your discretisation
cs    = (rho*g)**3*fs  # <<< this must be adjusted according to your discretisation
slope = 0.05
bmax = 1900.
def get_bedrock(xaxis):
    # here you put in your own equation that defines the bedrock
    bedrock = bmax - xaxis*slope
    return bedrock
def glacier_mass(hice,dx):
    return np.sum(hice)*dx*rho
#def glacier_height(hice)

def glacier_simulation(totL,dx,ntpy,elalist,elayear,dbdh,maxb):
    # Start calculations
    # constants that rely on input
    nx    = int(totL/dx)   #number of grid points
    dx    = totL/nx        #width of grid points     
    xaxis = np.linspace(0,totL,nx,False) + dx*0.5
    xhaxs = np.linspace(dx,totL,nx-1,False) / 1000.
    bedrock = get_bedrock(xaxis)
    dt    = 365.*86400./ntpy # in seconds!
    hice   = np.zeros(nx)    # ice thickness
    dhdx   = np.zeros(nx)    # the local gradient of h
    fluxd  = np.zeros(nx+2)  # this will be the flux per second!!!!
    fluxs  = np.zeros(nx+2)  # this will be the flux per second!!!!
    dhdtif = np.zeros(nx)    # change in ice thickness due to the ice flux, per second
    smb    = np.zeros(nx)    # surface accumulation
    # preparations for the ela-selection
    # elaswch is a list of time steps on which a new ela value should be used.
    nyear    = int(np.sum(elayear))
    if np.size(elalist) != np.size(elayear):
        print("the arrays of elalist and elayear does not have the same length!")
        exit()
    else:
        elaswch = np.zeros(np.size(elalist))
        for i in range(0,np.size(elaswch)-1):
            elaswch[i+1] = elaswch[i] + (elayear[i]*ntpy) #adds number of years * time steps per year to the list of steps
        ela     = elalist[0]


    print("Run model for {0:3d} years".format(nyear))
    hice_l=np.zeros((ntpy*nyear + 1,1))
    mass_l=np.zeros((ntpy*nyear + 1,1))
    for it in range(ntpy*nyear + 1):
        h = hice + bedrock
        dhdx[:-1]  = (h[1:]-h[:-1])/dx
        fluxd[1:-2] = cd * dhdx[:-1]**3 * ( ((hice[1:])+(hice[:-1])) * 0.5 )**5
        fluxs[1:-2] = cs * dhdx[:-1]**3 * ( ((hice[1:])+(hice[:-1])) * 0.5 )**3
        dhdtif[:]  = (fluxd[1:-1]-fluxd[:-2] + fluxs[1:-1]-fluxs[:-2])/dx
        if it%ntpy == 0:
            # lists the elements of elaswch that are equal or smaller than it
            [ielanow] = np.nonzero(elaswch<=it) 
            # the last one is the current ela
            ela       = elalist[ielanow[-1]]        
        smb[:] = (h-ela)*dbdh
        smb[:] = np.where(smb>maxb, maxb, smb) 
        hice   += smb/ntpy + dt*dhdtif
        hice[:] = np.where(hice<0., 0., hice) # remove negative ice thicknesses
        if ZeroFluxBoundary == False:
            hice[0] = hice[-1] = 0.

        hice_l[it]=[i for i, e in enumerate(hice) if e != 0][-1]
        mass_l[it]=glacier_mass(hice,dx)


        if it%(ndyfigure*ntpy) == 0:
            if StopWhenOutOfDomain:
                if hice[-1]>1.:
                    print("Ice at end of domain!")
                    exit()
    #Save data to dictionary
    simdict={}
    simdict['totL']=totL
    simdict['dx']=dx
    simdict['ntpy']=ntpy
    simdict['elalist']=elalist
    simdict['elayear']=elayear
    simdict['dbdh']=dbdh
    simdict['maxb']=maxb
    simdict['slope']=slope
    simdict['bmax']=bmax
    simdict['hice']=hice_l
    simdict['mass']=mass_l
    #np.save(simname,simdict,allow_pickle=True)
    return simdict
    
def closest(list, Number):
    # Find the closest value in list to Number - caution, if more than one x value for every y value,
    # can cause problems and pick the wrong one - adjust accordingly.
    aux = []
    for valor in list:
        aux.append(abs(Number-valor))

    return aux.index(min(aux))

def resp_time(data,num,diff,ts,mode):
    #From given data, find the response time for num different ELA jumps (separated by an adjustment time of diff), 
    #with ts timesteps per year, and direction of mass change (up/down) given by mode. Set up to not give the first rt,
    #as going from 0 to equilibrium is often not accurate.
    #Inputs:
    # - Data: list of data
    # - num: number of shifts in equilibrium level
    # - diff: time difference (years) between shifts of equilibrium level
    # - ts: numer of timesteps per year
    # - mode: whether data shows an upwards or downwards mass change (corresponding to a downwards or upwards ELA shift 
        #- keep it to one way per run, else it gets confused)
        
    eq_levels=[] #Level at which the glacier stabilizes
    rt=np.empty((num,1)) #Response times in number of time steps
    times=np.arange(0,diff*num*ts+1) #index of timesteps
    changes=np.arange(0,diff*num*ts+1,diff*ts) #At which indices the change in ELA occurs
    steps=changes[1:]-1 #Positions at which to take the eq.levels (1 ts before the change)
    ice_h=list(data) #turning the data into a useable list
    e_fold=1-1/np.exp(1) #efolding distance
       
    if mode=="up": #Corresponds to an increasing mass, or decreasing ELA
        
        for i in range(num):#first grab the equil. levels 
            eq_levels=np.append(eq_levels,data[int(steps[i])])
            
        for j in range(1,num): #process eq.levels 
            mass_diff=eq_levels[j]-eq_levels[j-1] #find diff in mass
            e_fold_m=e_fold*mass_diff+eq_levels[j-1] #find the "e folding mass" (m *0.63) and add to bottom level
            rt[j]=times[closest(ice_h,e_fold_m)]-changes[j] #find time this occurs and subtract time of ELA change
        
    elif mode == "down":# Corresponds to a decreasing mass, or increasing ELA
        
        jump=200000 #increase to prevent improper selection. "jumps" over first section of dataset
        ice_h_d=ice_h[jump:]
        
        for i in range(num):#first grab the equil. levels 
            eq_levels=np.append(eq_levels,data[int(steps[i])])
            
        for j in range(num-1): #process eq.levels 
            mass_diff=eq_levels[j]-eq_levels[j+1] #find diff in mass
            e_fold_m=eq_levels[j]-e_fold*mass_diff#find the "e folding mass" (m *0.63) and subtract from top level
            rt[j]=times[closest(ice_h_d,e_fold_m)]+jump-changes[j+1] #find time this occurs, subtract time of ELA change
            
    
    return rt/ts #returns eq.levels to check with graph above. rt is given in ts, need to divide to get years

def power_law(x, a, b):
    return a * x**-b