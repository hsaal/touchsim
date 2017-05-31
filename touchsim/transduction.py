import numpy as np
from scipy import linalg, interpolate, signal

from .constants import ihbasis

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
        Pdyn = linalg.solve(D,S1p.T,sym_pos=True)
    else:
        Pdyn = np.zeros(P.shape);
    return P, Pdyn

def block_solve(S0,D):
    nz = S0!=0
    # find similar lines to solve the linear system
    u,ia,ic = unique_rows(nz)
    unz = nz[ia,:] # unique non-zeros elements
    P = np.zeros(S0.shape)
    for ii in range(0,ia.size):
        lines = ic==ii    # lines of this block
        nzi = unz[ii,:]   # non-zeros elements
        ixgrid = np.ix_(lines,nzi)
        nzigrid = np.ix_(nzi,nzi)
        P[ixgrid] = linalg.solve(D[nzigrid],S0[ixgrid].T,sym_pos=True).T
    return P

def unique_rows(data):
    uniq, ia, ic = np.unique(data.view(data.dtype.descr * data.shape[1]),
        return_index=True,return_inverse=True)
    return uniq.view(data.dtype).reshape(-1, data.shape[1]), ia, ic

def circ_load_vert_stress(P,PLoc,PRad,AffLoc,AffDepth):
    AffDepth = np.atleast_2d(np.array(AffDepth))
    nsamp,npin = P.shape
    nrec = AffLoc.shape[0]

    x = AffLoc[:,0] - PLoc[:,0].T    # (npin,nrec)
    y = AffLoc[:,1] - PLoc[:,1].T    # (npin,nrec)
    z = np.dot(np.ones((npin,1)),AffDepth[:]) # (npin,nrec)

    r = np.hypot(x,y)

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

    s_z = eps * (J01 + XSI*J02)

    return s_z

def circ_load_dyn_wave(dynProfile,Ploc,PRad,Rloc,Rdepth,sfreq):
    # compute shift and decay only once for each unique x,y coord
    if Rloc.shape[0]>1:
        u,ia,ic = unique_rows(Rloc)
    else:
        ia = np.array([0])
        ic = np.array([0])

    nsamp = dynProfile.shape[1]
    npin = dynProfile.shape[0]
    nrec = ia.size

    dx = Ploc[:,0] - Rloc[ia,0].T    # (npin,nrec)
    dy = Ploc[:,1] - Rloc[ia,1].T    # (npin,nrec)
    dr = np.sqrt(dx**2 + dy**2)

    # delay (everything is synchronous under the probe)
    rdel = dr-PRad
    rdel[rdel<0] = 0.
    delay = rdel/8000. # 8000 is the wave velocity in mm/s

    # decay (=skin deflection decay given by Sneddon 1946)
    np.seterr(all="ignore")
    decay = 1./PRad/np.pi*np.arcsin(PRad/dr)
    np.seterr(all="warn")
    decay[dr<=PRad] = 1./2./PRad

    # construct interpolation functions for each pin and delays the propagation
    t = np.r_[1./sfreq:dynProfile.shape[1]/sfreq+1./sfreq:1./sfreq]
    udyn = np.zeros((nsamp,nrec))
    for jj in range(npin):
        loc_delays = t - delay[jj]
        F = interpolate.interp1d(t,dynProfile[jj,:].flatten(),
            bounds_error=False,fill_value=0.)
        delayed = np.reshape(F(loc_delays.flatten()),(nsamp,nrec))
        udyn += delayed*decay[jj]

    # copy results to all receptors at the same place
    udyn = udyn[:,ic]

    # z decay is 1/z^2
    udyn = udyn / (Rdepth**2)

    return udyn

def lif_neuron(aff,stimi,dstimi,srate):

    np.seterr(under="ignore")

    p = aff.parameters
    time_fac = srate/5000.

    # Make basis for post-spike current
    ih = ihbasis
    if time_fac!=1.:
        ih = interpolate.interp1d(
            np.r_[0.:0.0378:0.0002],ih,np.r_[0.:0.0378:0.0002/time_fac],kind='cubic')
    ih = np.dot(ih.T,p[10:12])

    if p[0]>0.:
        b,a = signal.butter(3,p[0]*4./(time_fac*1000.))
        stimi = np.atleast_2d(signal.lfilter(b,a,stimi.flatten())).T
        dstimi = np.atleast_2d(signal.lfilter(b,a,dstimi.flatten())).T

    ddstimi = np.r_[np.diff(dstimi,axis=0),np.zeros((1,1))]*time_fac

    s_all = np.c_[stimi,-stimi,dstimi,-dstimi,ddstimi,-ddstimi]
    s_all[s_all<0.] = 0.

    tau = p[9]     # decay time const
    vleak = 0.     # resting potential

    Iinj = s_all[:,0]*p[1] + s_all[:,1]*p[2] + s_all[:,2]*p[3] + s_all[:,3]*p[4] \
        + s_all[:,4]*p[5] + s_all[:,5]*p[6]

    if aff.noisy:
       Iinj += p[8]*np.random.standard_normal(Iinj.shape)

    if p[7]>0.:
        Iinj = p[7]*Iinj/(p[7]+np.abs(Iinj))
        Iinj[np.isnan(Iinj)] = 0.

    vthr = 1.   # threshold potential
    vr = 0.     # reset potential

    slen = Iinj.shape[0]
    nh = ih.shape[0]
    ih_counter = nh

    Vmem = vleak*np.ones((slen,1))
    Sp = np.zeros((slen,1))

    for ii in range(1,slen):  # Outer loop: 1 iter per time bin of input

        if ih_counter==nh:
            Vmem[ii] =  Vmem[ii-1] + (-(Vmem[ii-1]-vleak)/tau + Iinj[ii])/time_fac
        else:
            Vmem[ii] =  Vmem[ii-1] + (-(Vmem[ii-1]-vleak)/tau + Iinj[ii] + ih[ih_counter])/time_fac
            ih_counter += 1

        if Vmem[ii]>vthr and ih_counter>5*time_fac:
            Sp[ii] = 1.
            Vmem[ii] = vr
            ih_counter = 0

    spikes = np.flatnonzero(Sp)/srate + aff.parameters[12]/1000.

    np.seterr(under="warn")

    return spikes
