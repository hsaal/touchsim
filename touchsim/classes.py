import numpy as np
import random
try:
    import holoviews as hv
except:
    hv = None

from .transduction import skin_touch_profile, circ_load_vert_stress,\
    circ_load_dyn_wave, lif_neuron
from . import constants

class Afferent(object):

    affclasses = ['SA1','RA','PC']
    affdepths = constants.affdepths
    affparams = constants.affparams
    affcol = constants.affcol

    def __init__(self,affclass,**args):
        self.affclass = affclass
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

    def response(self,stim):
        strain, udyn, fs = stim.propagate(self)
        r = lif_neuron(self,strain,udyn,fs)
        return r

class AfferentPopulation(object):

    def __init__(self,*afferents):
        self.afferents = list(afferents)

    def num(self):
        return len(self.afferents)

    @property
    def affclass(self):
        return list(map(lambda x:x.affclass,self.afferents))

    @property
    def location(self):
        return np.asarray(list(map(lambda x:x.location.flatten(),self.afferents)))

    @property
    def depth(self):
        return np.asarray(list(map(lambda x:x.depth,self.afferents)))

    def find(self,affclass):
        return list(map(lambda x:x.affclass==affclass,self.afferents))

    def response(self,stim):
        r = list(map(lambda a:a.response(stim),self.afferents))
        return Response(self,stim,r)

    @ property
    def disp(self):
        if hv is None:
            return None
        return hv.NdOverlay({a:hv.Points(constants.coord2plot(self.location[self.find(a),:]))\
            (style=dict(color=Afferent.affcol[a])) for a in Afferent.affclasses})

class Stimulus:

    def __init__(self,**args):
        self.trace = np.atleast_2d(args.get('trace',np.array([[]])))
        self.location = np.atleast_2d(args.get('location',np.array([[0., 0.]])))
        self.fs = args.get('fs',1000.)
        self.pin_radius = args.get('pin_radius',.05)
        self.compute_profile()

    @property
    def duration(self):
        return self.trace.shape[1]/self.fs

    @property
    def time(self):
        return np.linspace(0.,self.duration,self.trace.shape[1])

    def disp(self,grid=False):
        if hv is None:
            return None
        d = {i:hv.Curve((self.time,self.trace[i]))
             for i in range(self.trace.shape[0])}
        if grid:
            return hv.NdLayout(d)
        else:
            return hv.NdOverlay(d)

    def compute_profile(self):
        self.profile, self.profiledyn = skin_touch_profile(
            self.trace,self.location,self.fs,self.pin_radius)

    def propagate(self,aff):
        stat_comp = circ_load_vert_stress(
            self.profile,self.location,self.pin_radius,aff.location,aff.depth)
        dyn_comp = circ_load_dyn_wave(
            self.profiledyn,self.location,self.pin_radius,aff.location,aff.depth,self.fs)
        return stat_comp, dyn_comp, self.fs

class Response:
    def __init__(self,a,s,r):
        self.aff = a
        self.stim = s
        self.spikes = r

    @property
    def duration(self):
        return self.stim.duration

    def rate(self):
        return (np.atleast_2d(np.array(list(map(lambda x:x.size, self.spikes))))/self.duration).T

    def psth(self,bin_width=10):
        bins = np.r_[0:self.duration+bin_width/1000.:bin_width/1000.]
        return np.array(list(map(lambda x:np.histogram(x,bins=bins)[0],self.spikes)))

    def disp_spikes(self):
        if hv is None:
            return None
        idx = [i for i in range(len(self.spikes)) if len(self.spikes[i]>0)]
        return hv.NdOverlay({i: hv.Spikes(self.spikes[idx[i]], kdims=['Time'])\
            (plot=dict(position=0.1*i),
            style=dict(color=Afferent.affcol[self.aff.afferents[idx[i]].affclass]))\
            for i in range(len(idx))})(plot=dict(yaxis='bare'))

    def disp_spatial(self,bin_width=10):
        if hv is None:
            return None
        if np.isinf(bin_width):
            r = self.rate()
        else:
            r = self.psth(bin_width)
        hm = dict()
        for t in range(r.shape[1]):
            hm[t] = hv.NdOverlay({a:hv.Points(np.concatenate(
                [constants.coord2plot(self.aff.location[self.aff.find(a),:]),
                r[self.aff.find(a),t:t+1]],axis=1),vdims=['Firing rate'])\
                (style=dict(color=Afferent.affcol[a])) for a in Afferent.affclasses})
        return hv.HoloMap(hm)
