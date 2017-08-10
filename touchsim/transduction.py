import numpy as np
from scipy import interpolate,signal
from numba import guvectorize,float64,boolean

from .constants import ihbasis

def check_pin_radius(loc,rad):
    if loc.shape[0]>1:
        if loc.shape[0]>=3:
            dx = loc[:,0:1] - loc[:,0:1].T
            dy = loc[:,1:2] - loc[:,1:2].T
            dist = np.sqrt(dx**2. + dy**2)
            dist[dist==0] = np.nan
            return np.amin(dist)/2.
        elif loc.shape[0]==2:
            return np.sqrt(np.sum((loc[0]-loc[1])**2))/2.
    else:
        return rad

def skin_touch_profile(S0,xy,samp_freq,ProbeRad):
    S0 = S0.T # hack, needs to be fixed
    s = S0.shape

    E = 0.05
    nu = 0.4

    x=xy[:,0]
    y=xy[:,1]

    R = np.sqrt((np.tile(x,(x.size,1))-np.tile(x,(x.size,1)).T)**2. \
        + (np.tile(y,(y.size,1))-np.tile(y,(y.size,1)).T)**2.)

    # flat cylinder indenter solution from (SNEDDON 1946):
    np.seterr(all="ignore")
    D = (1.-nu**2.)/np.pi/ProbeRad * np.arcsin(ProbeRad/R)/E
    np.seterr(all="warn")
    D[R<=ProbeRad] = (1.-nu**2.)/2./ProbeRad/E

    S0neg = S0<0
    absS0 = np.abs(S0)

    P = np.zeros(s)
    prevS0 = np.zeros(s)
    count=0
    # iterative contact-detection algorithm
    while count==0 or P[P<0].size>0:
        absS0[P<0] = 0.
        count += 1
        # only work on changed (and nonzeros) line
        diffl = np.sum(absS0-prevS0,axis=1) != 0.
        S0loc = absS0[diffl,:]
        P[diffl,:] = block_solve(S0loc,D)
        prevS0 = absS0.copy()

    # correct for the hack
    P[S0neg] = -P[S0neg]

    # actual skin profile under the pins
    S1 = np.dot(P,D)

    # time derivative of deflection profile
    # assumes same distribution of pressure as in static case
    # proposed by BYCROFT (1955) and confirmed by SCHMIDT (1981)
    if s[0]>1:
        # compute time derivative
        S1p = (np.r_[S1[1:,:], np.nan*np.ones((1,S1.shape[1]))] \
            - np.r_[np.nan*np.ones((1,S1.shape[1])), S1[0:-1,:]]) / 2. * samp_freq;
        S1p[0,:] = S1p[1,:]
        S1p[-1,:] = S1p[-2,:]
        # linsolve
        Pdyn = np.linalg.solve(D,S1p.T)
    else:
        Pdyn = np.zeros(P.shape);
    return P, Pdyn

def block_solve(S0,D):
    nz = S0!=0
    # do clever packing to speed up unique_rows
    if nz.shape[1]<128:
        packed = np.packbits(nz,axis=1)
    else:
        nz_ext = nz
        add = nz.shape[1] % 64
        if add>0:
            nz_ext = np.concatenate((nz,
                np.zeros((nz.shape[0],64-add),dtype=np.bool)),axis=1)
        packed = np.packbits(nz_ext,axis=1).view(np.uint64)

    # find similar lines to solve the linear system
    u,ia,ic = np.unique(packed,axis=0,return_index=True,return_inverse=True)
    unz = nz[ia,:] # unique non-zeros elements
    P = np.zeros(S0.shape)
    for ii in range(0,ia.size):
        lines = ic==ii    # lines of this block
        nzi = unz[ii,:]   # non-zeros elements
        ixgrid = np.ix_(lines,nzi)
        nzigrid = np.ix_(nzi,nzi)
        P[ixgrid] = np.linalg.solve(D[nzigrid],S0[ixgrid].T).T
    return P

def circ_load_vert_stress(P,PLoc,PRad,AffLoc,AffDepth):
    AffDepth = np.atleast_2d(np.array(AffDepth))
    nsamp,npin = P.shape
    nrec = AffLoc.shape[0]

    x = AffLoc[:,0:1] - PLoc[:,0:1].T    # (npin,nrec)
    y = AffLoc[:,1:2] - PLoc[:,1:2].T    # (npin,nrec)
    z = np.dot(np.ones((npin,1)),AffDepth) # (npin,nrec)

    r = np.hypot(x,y).T

    # Pressure stress matrix (r,t,z)  (SNEDDON 1946)
    XSI = z/PRad
    RHO = r/PRad

    rr = np.sqrt(1.+XSI**2.)
    R = np.sqrt((RHO**2. + XSI**2. - 1.)**2. + 4.*XSI**2.)
    theta = np.arctan(1./XSI)
    phi = np.arctan2(2.*XSI,(RHO**2. + XSI**2. -1.))

    J01 = np.sin(phi/2.) / np.sqrt(R)
    J02 = rr * np.sin(3./2.*phi - theta) / R**(3./2.)

    # Pressure rotated stress matrix (x,y,z)
    eps = P/2./PRad/PRad/np.pi

    s_z = np.dot(eps,(J01 + XSI*J02))

    return s_z

def circ_load_dyn_wave(dynProfile,Ploc,PRad,Rloc,Rdepth,sfreq,sur):
    # compute shift and decay only once for each unique x,y coord
    if Rloc.shape[0]>1:
        u,ia,ic = np.unique(Rloc,axis=0,return_index=True,return_inverse=True)
    else:
        ia = np.array([0])
        ic = np.array([0])

    nsamp = dynProfile.shape[1]
    npin = dynProfile.shape[0]
    nrec = ia.size

    dr = sur.distance(Ploc,Rloc[ia,:])

    # delay (everything is synchronous under the probe)
    rdel = dr-PRad
    rdel[rdel<0.] = 0.
    delay = np.atleast_2d(rdel/8000.) # 8000 is the wave velocity in mm/s

    # decay (=skin deflection decay given by Sneddon 1946)
    np.seterr(all="ignore")
    decay = 1./PRad/np.pi*np.arcsin(PRad/dr)
    np.seterr(all="warn")
    decay[dr<=PRad] = 1./2./PRad

    udyn = add_delays(delay.T,decay.T,dynProfile,sfreq)

    udyn = udyn.T
    # copy results to all receptors at the same place
    udyn = udyn[:,ic]

    # z decay is 1/z^2
    udyn = udyn / (Rdepth**2)

    return udyn

@guvectorize([(float64[:],float64[:],float64[:,:],float64[:],float64[:])],
    '(m),(m),(m,n),()->(n)',nopython=True,target='parallel')
def add_delays(delay,decay,dynProfile,sfreq,udyn):
    for i in range(udyn.shape[0]):
        udyn[i] = 0;
    for jj in range(dynProfile.shape[0]):
        delay_idx = int(np.rint(delay[jj]*sfreq[0]))
        if delay_idx>0:
            for i in range(delay_idx,dynProfile.shape[1]):
                udyn[i] += dynProfile[jj,i-delay_idx]*decay[jj]
        else:
            for i in range(dynProfile.shape[1]):
                udyn[i] += dynProfile[jj,i]*decay[jj]

def lif_neuron(aff,stimi,dstimi):
    srate = 5000. # fixed sampling frequency

    stimi = stimi.T
    dstimi = dstimi.T

    p = np.atleast_2d(aff.parameters)

    # Make basis for post-spike current
    ih = np.dot(p[:,10:12],ihbasis)

    uq,ia,ic = np.unique(np.atleast_2d(aff.gid),axis=0,
        return_index=True,return_inverse=True)
    for i in range(uq.shape[0]):
        bfilt,afilt = signal.butter(3,p[ia[i],0]*4./1000.)
        stimi[ic==i] = signal.lfilter(bfilt,afilt,stimi[ic==i],axis=1)
        dstimi[ic==i] = signal.lfilter(bfilt,afilt,dstimi[ic==i],axis=1)

    Iinj = weight_inputs(p,stimi,dstimi)

    Vmem = np.zeros(Iinj.shape)
    Sp = lif_sub(Vmem,Iinj,ih,p,aff.noisy)

    spikes = []
    for i in range(len(aff)):
        spikes.append(np.flatnonzero(Sp[i])/srate + p[i,12]/1000. + 1./srate)

    return spikes

@guvectorize([(float64[:],float64[:],float64[:],float64[:])],
    '(m),(n),(n)->(n)',nopython=True,target='parallel')
def weight_inputs(p,stimi,dstimi,Iinj):
    for i in range(stimi.shape[0]):
        if np.sign(stimi[i])>=0:
            Iinj[i]  = p[1]*stimi[i]
        else:
            Iinj[i]  = -p[2]*stimi[i]

        if np.sign(dstimi[i])>=0:
            Iinj[i]  += p[3]*dstimi[i]
        else:
            Iinj[i]  += -p[4]*dstimi[i]

        ddstimi = (dstimi[i+1 % stimi.shape[0]]-dstimi[i])
        if np.sign(ddstimi)>=0:
            Iinj[i]  += p[5]*ddstimi
        else:
            Iinj[i]  += -p[6]*ddstimi

@guvectorize([(float64[:],float64[:],float64[:],float64[:],boolean[:],
    float64[:])],'(n),(n),(m),(o),()->(n)',nopython=True,target='parallel')
def lif_sub(Vmem,Iinj,ih,p,noisy,Sp):
    if noisy[0]:
        Iinj += p[8]*np.random.standard_normal(Iinj.shape)

    tau = p[9]
    if p[7]>0.:
        Iinj = p[7]*Iinj/(p[7]+np.abs(Iinj))
        Iinj[np.isnan(Iinj)] = 0.

    nh = ih.size
    ih_counter = nh
    for ii in range(Vmem.size):

        if ih_counter==nh:
            Vmem[ii] =  Vmem[ii-1] + (-(Vmem[ii-1])/tau + Iinj[ii])
        else:
            Vmem[ii] =  Vmem[ii-1] + (-(Vmem[ii-1])/tau + Iinj[ii] + ih[ih_counter])
            ih_counter += 1

        if Vmem[ii]>1. and ih_counter>5:
            Sp[ii] = 1
            Vmem[ii] = 0.
            ih_counter = 0
        else:
            Sp[ii] = 0
