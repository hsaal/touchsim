import numpy as np
import random
import warnings
from math import isclose
from scipy.signal import resample

from .transduction import skin_touch_profile, circ_load_vert_stress,\
    circ_load_dyn_wave, lif_neuron, check_pin_radius
from . import constants
from .surface import null_surface

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
        self.delay = args.get('delay',False)
        self.surface = args.get('surface',null_surface)

        # set afferent depth
        if self.depth is None:
            self.depth = Afferent.affdepths.get(self.affclass)

        # set afferent parameters
        p = Afferent.affparams.get(self.affclass)
        if self.idx is None:
            self.idx = random.randint(0,p.shape[0]-1)
        self.parameters = p[self.idx].copy()

        if not self.delay:
            self.parameters[12] = 0.;

    def __len__(self):
        return 1

    @property
    def affclass(self):
        return self._affclass

    @affclass.setter
    def affclass(self,affclass):
        if not affclass in Afferent.affdepths.keys():
            raise IOError("Afferent class must be SA1, RA, or PC")
        self._affclass = affclass

    @property
    def gid(self):
        return np.asarray([Afferent.affclasses.index(self.affclass), self.idx])

    def __add__(self,other):
        if type(other) is Afferent:
            return AfferentPopulation(self,other)
        elif type(other) is AfferentPopulation:
            return other.__add__(self)
        else:
            RuntimeError("Can only add elements of type Afferent or AfferentPopulation.")
        return self

    def response(self,stim):
        strain, udyn, fs = stim.propagate(self)
        if not isclose(fs,5000.):
            strain = resample(strain,int(round(strain.shape[0]/fs*5000.)))
            udyn = resample(udyn,int(round(udyn.shape[0]/fs*5000.)))
        r = lif_neuron(self,strain,udyn)
        return Response(AfferentPopulation(self),stim,r)

class AfferentPopulation(object):

    def __init__(self,*afferents,**args):
        self.afferents = list(afferents)

        self.surface = args.pop('surface',null_surface)

        broadcast = aflag = False
        for key in args:
            v = args[key]
            if type(v) is list:
                n = len(v)
            elif type(v) is np.ndarray:
                n = v.shape[0]
            else:
                continue

            if n>1:
                b_key = key
                if b_key == 'affclass':
                    aflag = True
                b_val = args.pop(key)
                broadcast = True
                break

        if broadcast:
            for i in b_val:
                if aflag:
                    self.afferents.append(Afferent(i,**args))
                else:
                    self.afferents.append(Afferent(**{b_key:i},**args))
        elif len(args)>0:
            self.afferents.append(Afferent(**args))

    def __str__(self):
        return 'AfferentPopulation with ' + str(len(self)) + ' afferents: ' +\
                str(sum(self.find('SA1'))) + ' SA1, ' + str(sum(self.find('RA'))) +\
                 ' RA, ' + str(sum(self.find('PC'))) + ' PC.'

    def __len__(self):
        return len(self.afferents)

    def __getitem__(self,idx):
        if type(idx) is int:
            return self.afferents[idx]
        elif type(idx) is slice:
            return AfferentPopulation(*self.afferents[idx])
        elif type(idx) is (list or np.array) and len(idx)>0:
            if type(idx[0]) is bool:
                idx, = np.nonzero(idx)
            if type(idx) is np.array:
                idx = idx.tolist()
            return AfferentPopulation(*[self.afferents[i] for i in idx])
        elif idx in Afferent.affclasses:
            return self[self.find(idx)]
        else:
            raise TypeError(
                "Indices must be integers, slices, lists, or affclass.")

    def __add__(self,other):
        a = AfferentPopulation()
        if type(other) is Afferent:
            a.afferents.append(other)
        elif type(other) is AfferentPopulation:
            a.afferents.extend(other)
        else:
            raise TypeError(
                "Can only add elements of type Afferent or AfferentPopulation.")
        return a

    def __iadd__(self,other):
        if type(other) is Afferent:
            self.afferents.append(other)
        elif type(other) is AfferentPopulation:
            self.afferents.extend(other.afferents)
        else:
            raise TypeError(
                "Can only add elements of type Afferent or AfferentPopulation.")
        return self

    def num(self):
        return len(self)

    @property
    def affclass(self):
        return list(map(lambda x:x.affclass,self.afferents))

    @affclass.setter
    def affclass(self,affclass):
        if len(affclass)!=len(self):
            raise RuntimeError(
                "Length of affclass vector must match number of afferents")
        for i,a in enumerate(self.afferents):
            a.affclass=affclass[i]

    @property
    def gid(self):
        return np.asarray(list(map(lambda x:x.gid,self.afferents)))

    @property
    def location(self):
        return np.asarray(list(map(lambda x:x.location.flatten(),
            self.afferents)))

    @property
    def depth(self):
        return np.asarray(list(map(lambda x:x.depth,self.afferents)))

    @property
    def parameters(self):
        return np.asarray(list(map(lambda x:x.parameters.flatten(),
            self.afferents)))

    @property
    def noisy(self):
        return np.asarray(list(map(lambda x:x.noisy,self.afferents)))

    def find(self,affclass):
        return list(map(lambda x:x.affclass==affclass,self.afferents))

    def response(self,stim):
        strain, udyn, fs = stim.propagate(self)
        if not isclose(fs,5000.):
            strain = resample(strain,int(round(strain.shape[0]/fs*5000.)))
            udyn = resample(udyn,int(round(udyn.shape[0]/fs*5000.)))
        r = lif_neuron(self,strain,udyn)
        return Response(self,stim,r)

class Stimulus(object):

    def __init__(self,**args):
        self.trace = np.atleast_2d(args.get('trace',np.array([[]])))
        self.location = np.atleast_2d(args.get('location',np.array([[0., 0.]])))
        self.fs = args.get('fs',1000.)
        self.pin_radius = args.get('pin_radius',.05)
        self.compute_profile()

    def __str__(self):
        return 'Stimulus with ' + str(self.location.shape[0]) +\
            ' pins and ' + str(self.duration) + ' s total duration.'

    @property
    def duration(self):
        return self.trace.shape[1]/self.fs

    @property
    def time(self):
        return np.linspace(0.,self.duration,self.trace.shape[1])

    def __iadd__(self,other):
        if type(other) is not Stimulus:
            raise RuntimeError("Can only add objects of type Stimulus.")
        if self.pin_radius!=other.pin_radius:
            warnings.warn("Overwriting pin_radius of second Stimulus object!")
        if self.fs!=other.fs:
            raise RuntimeError("Sampling frequencies must be the same.")
        if self.duration!=other.duration:
            raise RuntimeError("Stimulus durations must be the same.")

        self.trace = np.concatenate([self.trace,other.trace])
        self.location = np.concatenate([self.location,other.location])
        self.compute_profile()
        return self

    def compute_profile(self):
        new_radius = check_pin_radius(self.location,self.pin_radius)
        if self.pin_radius>new_radius:
            warnings.warn(
                "Pin radius too big and has been adjusted to %.1f" % new_radius)
            self.pin_radius = new_radius

        self._profile, self._profiledyn = skin_touch_profile(
            self.trace,self.location,self.fs,self.pin_radius)

    def propagate(self,aff):
        stat_comp = circ_load_vert_stress(
            self._profile,self.location,self.pin_radius,aff.location,aff.depth)
        dyn_comp = circ_load_dyn_wave(
            self._profiledyn,self.location,self.pin_radius,aff.location,
                aff.depth,self.fs,aff.surface)
        return stat_comp, dyn_comp, self.fs

class Response(object):
    def __init__(self,a,s,r):
        if len(a)!=len(r):
            RuntimeError("a and r need to have the same length.")

        self.aff = a
        self.stim = s
        self.spikes = r

    def __str__(self):
        return 'Response consisting of:\n* ' + self.aff.__str__() + '\n* ' +\
            self.stim.__str__() + '\n* ' + str(np.sum(self.psth(self.duration))) +\
            ' total spikes.'

    def __len__(self):
        return len(self.aff)

    def __getitem__(self,idx):
        if type(idx) is int:
            return Response(self.aff[idx],self.stim,self.spikes[idx])
        elif type(idx) is Afferent:
            return Response(idx,self.stim,self.spikes[self.aff.index(idx)])
        elif type(idx) is AfferentPopulation:
            return Response(idx,self.stim,
                [self.spikes[self.aff.index(a)] for a in idx])
        else:
            return Response(self.a[idx],self.stim,self.spikes[idx])

    @property
    def duration(self):
        return self.stim.duration

    def rate(self):
        return (np.atleast_2d(np.array(list(map(lambda x:x.size, self.spikes)))) \
            /self.duration).T

    def psth(self,bin_width=10):
        bins = np.r_[0:self.duration+bin_width/1000.:bin_width/1000.]
        return np.array(list(map(lambda x:np.histogram(x,bins=bins)[0],self.spikes)))
