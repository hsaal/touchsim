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
    ''' Creates single afferent object.
    '''
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

        if self.depth is None:   # Set afferent depth
            self.depth = Afferent.affdepths.get(self.affclass)

        p = Afferent.affparams.get(self.affclass)      # Set afferent parameters
        if self.idx is None:
            self.idx = random.randint(0,p.shape[0]-1)
        self.parameters = p[self.idx].copy()

        if not self.delay:
            self.parameters[12] = 0.;

    def __str__(self):
        return 'Afferent of class ' + str(self.affclass) + ' (model id:  ' +\
            str(self.gid) + ')'

    def __len__(self):
        return 1

    def __getitem__(self,idx):
        if idx is True or idx==0 or idx==self.affclass or \
            (type(idx) is (list or np.array) and len(idx)>0 and\
            (idx[0] is True or idx[0]==0 or idx[0]==self.affclass)):

            return AfferentPopulation(self)
        else:
            return AfferentPopulation()

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

    def find(self,affclass):
        return [self.affclass==affclass]

    def response(self,stim):
        assert type(stim) is Stimulus or type(stim[0]) is Stimulus,\
            "Argument needs to be Stimulus object or an iterable over Stimulus objects."

        try:
            s_iter = iter(stim)
        except:
            stim = [stim]
            s_iter = iter(stim)
        r = list()
        for s in s_iter:
            strain, udyn, fs = s.propagate(self)
            if not isclose(fs,5000.):
                strain = resample(strain,int(round(strain.shape[0]/fs*5000.)))
                udyn = resample(udyn,int(round(udyn.shape[0]/fs*5000.)))
            r.append(lif_neuron(self,strain,udyn))
        return Response(AfferentPopulation(self),stim,r)


class AfferentPopulation(object):
    ''' Creates an afferent population object.
    '''
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
        ''' _str_ creates a string representation of the object,
        shows the number of each afferent type in the afferent population.
        '''
        return 'AfferentPopulation with ' + str(len(self)) + ' afferent(s): ' +\
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
        assert type(stim) is Stimulus or type(stim[0]) is Stimulus,\
            "Argument needs to be Stimulus object or an iterable over Stimulus objects."

        try:
            s_iter = iter(stim)
        except:
            stim = [stim]
            s_iter = iter(stim)
        r = list()
        for s in s_iter:
            strain, udyn, fs = s.propagate(self)
            if not isclose(fs,5000.):
                strain = resample(strain,int(round(strain.shape[0]/fs*5000.)))
                udyn = resample(udyn,int(round(udyn.shape[0]/fs*5000.)))
            r.append(lif_neuron(self,strain,udyn))
        return Response(self,stim,r)


class Stimulus(object):
    ''' Creates a stimulus object.
    '''
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

    def __imul__(self,other):
        if type(other) is not Stimulus:
            raise RuntimeError("Can only add objects of type Stimulus.")
        if self.pin_radius!=other.pin_radius:
            warnings.warn("Overwriting pin_radius of second Stimulus object!")
        if self.fs!=other.fs:
            raise RuntimeError("Sampling frequencies must be the same.")

        if np.array_equal(self.location,other.location):
            self.trace = np.concatenate([self.trace,other.trace],axis=1)
            self._profile = np.concatenate([self.profile,other.profile],axis=1)
            self._profiledyn = np.concatenate([self.profiledyn,other.profiledyn],axis=1)
        else:
            self.trace = np.concatenate([np.concatenate(
                [self.trace,np.zeros((other.trace.shape[0],self.trace.shape[1]))]),
                np.concatenate(
                [np.zeros((self.trace.shape[0],other.trace.shape[1])),other.trace])],
                axis=1)
            self.location = np.concatenate([self.location,other.location])
            self._profile = np.concatenate([np.concatenate(
                [self._profile,np.zeros((other._profile.shape[0],self._profile.shape[1]))]),
                np.concatenate(
                [np.zeros((self._profile.shape[0],other._profile.shape[1])),other._profile])],
                axis=1)
            self._profiledyn = np.concatenate([np.concatenate(
                [self._profiledyn,np.zeros((other._profiledyn.shape[0],self._profiledyn.shape[1]))]),
                np.concatenate(
                [np.zeros((self._profiledyn.shape[0],other._profiledyn.shape[1])),other._profiledyn])],
                axis=1)
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
    ''' Creates a response object to stimulus.
    '''
    def __init__(self,a,s,r):
        if len(a)!=len(r):
            RuntimeError("a and r need to have the same length.")

        self.aff = a
        self.stim = s
        self._spikes = r

    def __str__(self):
        return 'Response consisting of:\n* ' + self.aff.__str__() + '\n* ' +\
            str(self.__len__()) + ' stimuli with ' + str(self.duration) +\
            ' s total duration.' +\
            '\n* ' + str(int(np.sum(self.rate())*self.duration)) + ' total spikes.'

    def __len__(self):
        return len(self.stim)

    def __getitem__(self,idx):
        if type(idx) is int:
            return Response(self.aff[idx],self.stim,self.spikes[idx])
        elif type(idx) is Afferent:
            return Response(idx,self.stim,self.spikes[self.aff.afferents.index(idx)])
        elif type(idx) is AfferentPopulation:
            return Response(idx,self.stim,
                [self.spikes[self.aff.afferents.index(a)] for a in idx])
        else:
            return Response(self.a[idx],self.stim,self.spikes[idx])

    @property
    def duration(self):
        d = 0
        for s in iter(self.stim):
            d += s.duration
        return d

    @property
    def durations(self):
        return [s.duration for s in iter(self.stim)]

    @property
    def spikes(self):
        if len(self)==1:
            return self._spikes[0]
        else:
            sp = [np.array([]) for i in range(len(self._spikes[0]))]
            cum_dur = 0.
            for s in range(len(self._spikes)):
                sp = [np.concatenate((sp[i],self._spikes[s][i]+cum_dur)) for i in range(len(self._spikes[0]))]
                cum_dur += self.stim[s].duration
            return sp

    def rate(self,sep=False):
        r = np.zeros((len(self.aff),len(self)))
        for i,s in enumerate(self._spikes):
            r[:,i:i+1] = (np.atleast_2d(np.array(list(map(lambda x:x.size, s)))) \
                /self.durations[i]).T
        if not sep:
            r = np.atleast_2d(np.mean(r,axis=1)).T
        return r

    def psth(self,bin=10):
        bins = np.r_[0:self.duration+bin/1000.:bin/1000.]
        return np.array(list(map(lambda x:np.histogram(x,bins=bins)[0],self.spikes)))
