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
    # at this point, the simulation is completed.        
    # the following is needed to make the animation