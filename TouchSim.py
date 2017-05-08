import numpy as np
import MechanoTransduction
import Constants
import random
import re
from matplotlib import path

class Afferent(object):

    affdepths = Constants.affdepths
    affparamsSA = Constants.affparamsSA
    affparamsRA = Constants.affparamsRA
    affparamsPC = Constants.affparamsPC
    affparams = Constants.affparams

    def __init__(self,affclass,**args):
        self._affclass = self.affclass = affclass
        self.location = np.atleast_2d(args.get('location',np.array([[0., 0.]])))
        self.depth = args.get('depth',None)
        self.idx = args.get('idx',None)
        self.noisy = args.get('noisy',True)
        self.delay = args.get('delay',True)

        # set afferent depth
        if self.depth is None:
            self.depth = Afferent.affdepths.get(self.affclass)

        # set afferent parameters
        p = Afferent.affparams.get(self.affclass)
        if self.idx is None:
            self.idx = random.randint(0,p.shape[0]-1)
        self.parameters = p[self.idx]

        if not self.delay:
            self.parameters[10] = 0.;

    @property
    def affclass(self):
        return self._affclass

    @affclass.setter
    def affclass(self,affclass):
        if not affclass in Afferent.affdepths.keys():
            raise IOError("Afferent class must be SA1, RA, or PC")
        self._affclass = affclass

    def iSA1(self):
        return self.affclass=='SA1'

    def iRA(self):
        return self.affclass=='RA'

    def iPC(self):
        return self.affclass=='PC'

    def response(self,stim):
        strain, udyn, fs = stim.propagate(self)
        r = MechanoTransduction.lif_neuron(self,strain,udyn,fs)
        return r

class AfferentPopulation(object):

    def __init__(self,*afferents):
        self.afferents = list(afferents)

    def num(self):
        return len(self.afferents)

    def affclass(self):
        return list(map(lambda x:x.affclass,self.afferents))

    def iSA1(self):
        return list(map(lambda x:x.iSA1(),self.afferents))

    def iRA(self):
        return list(map(lambda x:x.iRA(),self.afferents))

    def iPC(self):
        return list(map(lambda x:x.iPC(),self.afferents))

    def location(self):
        return np.asarray(list(map(lambda x:x.location.flatten(),self.afferents)))

    def response(self,stim):
        r = []
        for a in self.afferents:
            r.append(a.response(stim))
        return r

class Stimulus:

    def __init__(self,**args):
        self.trace = np.atleast_2d(args.get('trace',np.array([[]])))
        self.location = np.atleast_2d(args.get('location',np.array([[0., 0.]])))
        self.fs = args.get('fs',1000.)
        self.pin_radius = args.get('pin_radius',.05)
        self.compute_profile()

    def duration(self):
        return self.trace.shape[1]/self.fs

    def compute_profile(self):
        self.profile, self.profiledyn = MechanoTransduction.skin_touch_profile(
            self.trace,self.location,self.fs,self.pin_radius)

    def propagate(self,aff):
        stat_comp = MechanoTransduction.circ_load_vert_stress(
            self.profile,self.location,self.pin_radius,aff.location,aff.depth)
        dyn_comp = MechanoTransduction.circ_load_dyn_wave(
            self.profiledyn,self.location,self.pin_radius,aff.location,aff.depth,self.fs)
        return stat_comp, dyn_comp, self.fs

def affpop_single_models(**args):
    affclass = args.get('affclass',Afferent.affparams.keys())
    a = AfferentPopulation()
    for t in affclass:
        for i in range(Afferent.affparams.get(t).shape[0]):
            a.afferents.append(Afferent(t,idx=i))
    return a

def affpop_hand(**args):
    affclass = args.get('affclass',Afferent.affparams.keys())
    region = args.get('region',None)
    if region is None:
        idx = range(20)
    else:
        match = re.findall('[dDpPwWmMdDfFtT]\d?',region)
        idx = [i for i,x in enumerate(Constants.regionprop_tags) if x[0]==match[0]]
        if len(match)>1:
            idx = set(idx).intersection([i for i,x in enumerate(Constants.regionprop_tags) if x[1]==match[1]])

    affpop = AfferentPopulation()
    for a in affclass:
        for i in idx:
            dens = np.sqrt(Constants.density[(a,
                Constants.regionprop_tags[i][2])])/10./Constants.pxl_per_mm
            b = Constants.regionprop_boundingbox[i,:]
            xy = np.mgrid[b[0]:b[0]+b[2]+1./dens:1./dens,b[1]:b[1]+b[3]+1./dens:1./dens]
            xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
            xy += np.random.randn(xy.shape[0],xy.shape[1])/dens/5.
            p = path.Path(Constants.regionprop_boundary[i].T)
            ind = p.contains_points(xy);
            xy = xy[ind,:]

            xy -= Constants.orig
            xy = np.dot(xy,Constants.rot)/Constants.pxl_per_mm
            for l in range(xy.shape[0]):
                affpop.afferents.append(Afferent(a,location=xy[l,:]))
    return affpop


def stim_sine(**args):
    freq = np.array(args.get('freq',200.))
    amp = np.array(args.get('amp',.02*np.ones(freq.shape)))
    phase = np.array(args.get('phase',np.zeros(freq.shape)))
    len = args.get('len',1.)
    loc = np.array(args.get('loc',np.array([0.,0.])))
    fs = args.get('fs',5000.)
    ramp_len = args.get('ramp_len',.05)
    ramp_type = args.get('ramp_type','lin')
    pin_size = args.get('pin_size',.5)
    pre_indent = args.get('pre_indent',0.)
    pad_len = args.get('pad_len',0.)

    trace = np.zeros(int(fs*len))
    for f,a,p in zip(np.nditer(freq),np.nditer(amp),np.nditer(phase)):
        trace += a*np.sin(p*np.pi/180. \
            + np.linspace(0.,2.*np.pi*f*len,int(fs*len)))

    apply_ramp(trace,len=ramp_len,fs=fs)
    apply_pad(trace,len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,pin_radius=pin_size)

def stim_ramp(**args):
    amp = args.get('amp',1.)
    ramp_type = args.get('ramp_type','lin')
    len = args.get('len',1.)
    loc = np.array(args.get('loc',np.array([0.,0.])))
    fs = args.get('fs',5000.)
    ramp_len = args.get('ramp_len',.1)
    pin_size = args.get('pin_size',.5)
    pre_indent = args.get('pre_indent',0.)
    pad_len = args.get('pad_len',0.)

    trace = amp*np.ones(int(fs*len))
    apply_ramp(trace,len=ramp_len,fs=fs,ramp_type=ramp_type)
    apply_pad(trace,len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,pin_radius=pin_size)

def apply_ramp(trace,**args):
    len = args.get('len',.05)
    fs = args.get('fs',None)
    ramp_type = args.get('ramp_type','lin')
    if max(len,0.)==0.:
        return
    if fs is not None:
        len = round(len*fs)

    # apply ramp
    if ramp_type=='lin':
        trace[:len] *= np.linspace(0,1,len)
        trace[-len:] *= np.linspace(1,0,len)
    elif ramp_type=='sin' or ramp_type=='sine':
        trace[:len] *= np.cos(np.linspace(np.pi,2.*np.pi,len)/2.+.5)
        trace[-len:] *= np.cos(np.linspace(0.,np.pi,len)/2.+.5)
    else:
        raise IOError("ramp_type must be 'lin' or 'sin'")

def apply_pad(trace,**args):
    len = args.get('len',.05)
    fs = args.get('fs',None)
    if max(len,0.)==0.:
        return
    if fs is not None:
        len = round(len*fs)
    trace = np.concatenate((np.zeros(len),trace,np.zeros(len)))
