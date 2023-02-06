import numpy as np
import random
import warnings
from math import isclose
from scipy.signal import resample
from pathlib import Path
import json

from .transduction import skin_touch_profile, circ_load_vert_stress,\
    circ_load_dyn_wave, lif_neuron, check_pin_radius
from . import constants
from .surface import Surface, null_surface

class Afferent(object):
    """A single afferent, which can be placed on a surface and respond to tactile
    stimuli.
    """
    affclasses = constants.affclasses
    affdepths = constants.affdepths
    affparams = constants.affparams
    affcol = constants.affcol

    def __init__(self,affclass,**args):
        """Initializes an Afferent object.

        Args:
            affclass (string): Afferent class, one of 'SA1', 'RA', or 'PC'.

        Kwargs:
            location (array): Location of the afferent (default: [0,0]).
            noisy (bool): Injects noise into membrane potential (default: True).
            delay (bool) = Adds delays to mimic travel to neural recording site
                (default: False).
            surface (Surface object): The surface on which Afferent is located
                (default: null_surface).
            depth (float) = Depth of afferent in the skin (default: standard depth
                depening on afferent class).
            idx (int): ID number of neuron model (default: randomly chosen).
        """
        self.affclass = affclass
        self.location = np.atleast_2d(args.get('location',np.array([[0., 0.]])))
        self.depth = args.get('depth',None)
        self.idx = args.get('idx',None)
        self.noisy = args.get('noisy',True)
        self.delay = args.get('delay',False)
        self.surface = args.get('surface',null_surface)

        # Meta dictionary containing all data required to recreate exact afferent
        self.meta = args
        self.meta["type"] = affclass

        if self.depth is None:   # Set afferent depth
            self.depth = Afferent.affdepths.get(self.affclass)

        p = Afferent.affparams.get(self.affclass)      # Set afferent parameters
        if self.idx is None:
            self.idx = random.randint(0,p.shape[0]-1)
        self.parameters = p[self.idx].copy()

        if not self.delay:
            self.parameters[12] = 0.;

    def __str__(self):
        return 'Afferent of class ' + self.affclass + ' (gid: ' +\
            str(self.gid) + ').'

    def __len__(self):
        return 1

    @property
    def affclass(self):
        return self._affclass

    @affclass.setter
    def affclass(self,affclass):
        if not affclass in Afferent.affclasses:
            raise IOError("Afferent class must be one of " + \
                ", ".join(Afferent.affclasses) + ".")
        self._affclass = affclass

    @property
    def gid(self):
        return np.asarray([Afferent.affclasses.index(self.affclass), self.idx])

    @property
    def region(self):
        return self.surface.locate(self.location)

    def __add__(self,other):
        if type(other) is Afferent:
            return AfferentPopulation(self,other)
        elif type(other) is AfferentPopulation:
            return AfferentPopulation(self,*other.afferents)
        else:
            RuntimeError("Can only add elements of type Afferent or AfferentPopulation.")
        return self

    def response(self,stim):
        """Calculates the afferent's spiking response to a tactile stimulus.

        Args:
            stim (Stimulus object): The tactile stimulus.

        Returns:
            Response object.
        """
        return AfferentPopulation(self).response(stim)


class AfferentPopulation(object):
    """A population of afferents.
    """

    def __init__(self,*afferents,**args):
        """Initializes an AfferentPopulation object.

        Args:
            a1, a2, ... (Afferent): Afferent objects.

        Kwargs:
            surface (Surface object): The surface on which Afferent is located
                (default: a1.surface if set, otherwise null_surface).
        """
        self.afferents = list(afferents)
        if len(self.afferents)==0:
            sur = null_surface
        else:
            sur = self.afferents[0].surface
        self.surface = args.get('surface', sur)
        for a in self.afferents:
            a.surface = self.surface

    def __str__(self):
        return 'AfferentPopulation with ' + str(len(self)) + ' afferent(s): ' +\
                str(sum(self.find('SA1'))) + ' SA1, ' + str(sum(self.find('RA'))) +\
                 ' RA, ' + str(sum(self.find('PC'))) + ' PC.'

    def __len__(self):
        return len(self.afferents)

    def __getitem__(self,idx):
        if type(idx) is int:
            return self.afferents[idx]
        elif type(idx) is slice:
            return AfferentPopulation(*self.afferents[idx],surface=self.surface)
        elif (type(idx) is list and len(idx)>0) or (type(idx) is np.ndarray and idx.size>0):
            if type(idx[0]) is bool or type(idx[0]) is np.bool_:
                idx, = np.nonzero(idx)
            if type(idx) is np.ndarray:
                idx = idx.tolist()
            return AfferentPopulation(*[self.afferents[i] for i in idx],surface=self.surface)
        elif idx in Afferent.affclasses:
            return self[self.find(idx)]
        elif type(idx) is str:
            ridx = self.surface.tag2idx(idx)
            rr = self.region[1]
            bidx = np.zeros((len(self,)),dtype=np.bool_)
            for r in ridx:
                bidx = np.logical_or(bidx,np.array([rr_sub==r for rr_sub in rr]))
            return self[bidx]
        else:
            raise TypeError(
                "Indices must be integers, slices, lists, affclass, or surface region.")

    def __add__(self,other):
        a = AfferentPopulation()
        a.afferents = list(self.afferents)
        if type(other) is Afferent:
            return AfferentPopulation(*self.afferents,other)
        elif type(other) is AfferentPopulation:
            return AfferentPopulation(*self.afferents,*other.afferents)            
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
    def region(self):
        return self.surface.locate(self.location)

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
        """Finds afferents in the population of a given class.

        Args:
            affclass (string): Afferent class (e.g. 'SA1').

        Returns:
            List of True/False values.
        """
        return list(map(lambda x:x.affclass==affclass,self.afferents))

    def response(self,stim):
        """Calculates the afferent population's spiking response to a tactile
        stimulus.

        Args:
            stim (Stimulus object): The tactile stimulus.

        Returns:
            Response object.
        """
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # suppress underflow warnings
                r.append(lif_neuron(self,strain,udyn))
        return Response(self,stim,r)

    def save_population(self, filename:str, make_dirs:bool=True, overwrite:bool=True) -> None:
        """ Saves afferent population into a json file

        Method creates a dictionary containing the relevant meta information of each afferent and surface within population.
        Errors are raised by path.parent.mkdir function if make_dirs and overwrite bools are not correctly set for your setup.

        Args:
            filename (string): Path to json file. Must include .json suffix
            make_dirs (bool) : Indicate whether directories on output path should be created (default: True)
            overwrite (bool) : Allows overwrite of file if it already exists on output path (default: True)
        Returns:
            None
        """
        path = Path(filename)

        # Make parent directories if allowed by make_dirs and overwrite bools
        path.parent.mkdir(parents=make_dirs, exist_ok=overwrite)

        population_dict = {}

        # Prepare afferents for json
        for idx, afferent in enumerate(self.afferents):
            afferent.meta.pop("surface", None)
            population_dict[idx] = afferent.meta

        # Prepare surface for json
        # Problem lies in the density subdict using tuples as keys - json does not like this
        # Convert tuple key to string with "/" seperator
        density_dict = {f"{key[0]}/{key[1]}" if type(key) is tuple else key:_ for key, _ in self.surface.meta["density"].items()}

        surface_dict = self.surface.meta.copy()
        surface_dict["density"] = density_dict
        
        population_dict["surface"] = surface_dict

        # Clean up arrays to make json serializable
        for component, params in population_dict.items():
            component_dict = {_: param.tolist() if type(param) is np.ndarray else param for _, param in params.items()}
            population_dict[component] = component_dict

        with path.open("w") as out:
            json.dump(population_dict, out)

    @classmethod
    def load_population(cls, filename:str) -> None:
        """ Loads an afferent population from a json file. To be used in conjunction with save_population method

        Arguments:
            filename: Path to json file. Must include .json suffix
        Returns:
            None:
        """
        path = Path(filename)
        afferents = []
        
        with path.open("r") as input:
            population_data = json.load(input)

        # Get surface for population, reset lists back to arrays
        surface_params = population_data.pop("surface")
        surface_params["orig"] = np.array(surface_params["orig"])
        # Gross converting of keys back to tuples
        density_dict = surface_params["density"]
        for key in list(density_dict.keys()):
            tuple_key = tuple(key.split("/"))
            density_dict[tuple_key] = density_dict.pop(key)

        # Build afferent population
        for _, params in population_data.items():
            t = params.pop("type")
            if "location" in params:
                params["location"] = np.array(params["location"])
            afferents.append(Afferent(t, **params))

        surface = Surface(**surface_params)
        afferents = list(afferents)

        return cls(*afferents, surface=surface)

class Stimulus(object):
    """A tactile stimulus.
    """

    def __init__(self,**args):
        """Initializes a Stimulus object.

        Kwargs:
            trace (NxT array) = np.atleast_2d(args.get('trace',np.array([[]])))
            location (Nx2 array) = np.atleast_2d(args.get('location',np.array([[0., 0.]])))
            fs (float): Sampling frequency (default: 1000.).
            pin_radius (float): Pin radius in mm (default: 0.05).
        """
        self.trace = np.atleast_2d(args.get('trace',np.array([[]])))
        self.location = np.atleast_2d(args.get('location',np.array([[0., 0.]])))
        self.fs = args.get('fs',1000.)
        self.pin_radius = args.get('pin_radius',.05)
        self.compute_profile()
        self.meta = args

    def __str__(self):
        return 'Stimulus with ' + str(self.location.shape[0]) +\
            ' pins and ' + str(self.duration) + ' s total duration.'

    def __len__(self):
        return self.location.shape[0]

    @property
    def duration(self):
        return self.trace.shape[1]/self.fs

    @property
    def time(self):
        return np.linspace(0.,self.duration,self.trace.shape[1])

    def __iadd__(self,other):
        if type(other) is not Stimulus:
            raise TypeError("Can only add objects of type Stimulus.")
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
        """Computes surface profile over time. This method is executed
        automatically whenever the 'trace' property changes.
        """
        new_radius = check_pin_radius(self.location,self.pin_radius)
        if self.pin_radius>new_radius:
            warnings.warn(
                "Pin radius too big and has been adjusted to %.1f" % new_radius)
            self.pin_radius = new_radius

        self._profile, self._profiledyn = skin_touch_profile(
            self.trace,self.location,self.fs,self.pin_radius)

    def propagate(self,aff):
        """Propagates the stimulus to specific afferent locations.

        Args:
            aff (Afferent or AfferentPopulation object): The afferent location(s)
                the stimulus is propagated to.

        Returns:
            Tuple consisting of the static mechanical component, the dynamic
            mechanical component, and the sampling rate.
        """
        stat_comp = circ_load_vert_stress(
            self._profile,self.location,self.pin_radius,aff.location,aff.depth)
        dyn_comp = circ_load_dyn_wave(
            self._profiledyn,self.location,self.pin_radius,aff.location,
                aff.depth,self.fs,aff.surface)
        return stat_comp, dyn_comp, self.fs

    def save_stimulus(self, filename:str, make_dirs:bool=True, overwrite:bool=True) -> None:
        """ Saves stimulus into a json file

        Method creates a dictionary containing the relevant meta information of recreating a stimulus.
        Errors are raised by path.parent.mkdir function if make_dirs and overwrite bools are not correctly set for your setup.

        Args:
            filename (string): Path to json file. Must include .json suffix
            make_dirs (bool) : Indicate whether directories on output path should be created (default: True)
            overwrite (bool) : Allows overwrite of file if it already exists on output path (default: True)
        Returns:
            None
        """
        path = Path(filename)

        # Make parent directories if allowed by make_dirs and overwrite bools
        path.parent.mkdir(parents=make_dirs, exist_ok=overwrite)

        new_params = {key: param.tolist() if type(param) is np.ndarray else param for key, param in self.meta.items()}

        with path.open("w") as out:
            json.dump(new_params, out)

    @classmethod
    def load_stimulus(cls, filename:str) -> None:

        """ Loads a stimulus from a json file. To be used in conjunction with save_stimulus method

        Arguments:
            filename: Path to json file. Must include .json suffix
        Returns:
            None:
        """
        path = Path(filename)
        
        with path.open("r") as input:
            stimulus_data = json.load(input)

        params = {key: np.array(param) if type(param) is list else param for key, param in stimulus_data.items()}

        return cls(**params)


class Response(object):
    """A Response by an AfferentPopulation to a Stimulus.
    """

    def __init__(self,a,s,r):
        """Initializes a Response object.

        Args:
            a (AfferentPopulation): The population of responding afferents
            s (Stimulus or list): The Stimulus (or list of Stimuli) that the
                afferents are responding to.
            r (list): List of arrays containing the spike times for each afferent,
                contained in another list with entries for each Stimulus object.

        Note:
            Response objects are created by calling the response method of an
            Afferent or AfferentPopulation object; there should be little need to
            call the constructor manually.
        """
        assert len(s)==len(r)
        assert len(a)==len(r[0])

        self.aff = a
        self.stim = s
        self._spikes = r

    def __str__(self):
        return 'Response consisting of:\n* ' + self.aff.__str__() + '\n* ' +\
            str(len(self.stim)) + ' stimuli with ' + str(self.duration) +\
            ' s total duration.' +\
            '\n* ' + str(int(np.sum(self.rate())*self.duration)) + ' total spikes.'

    def __len__(self):
        return len(self.aff)

    def __getitem__(self,idx):
        if type(idx) is Afferent:
            a = AfferentPopulation(idx)
            ii = [self.aff.afferents.index(idx)]
        elif type(idx) is AfferentPopulation:
            a = idx
            ii = [self.aff.afferents.index(a) for a in idx]
        else:
            raise TypeError("Index must be Afferent or AfferentPopulation.")

        r = list()
        for i in range(len(self.stim)):
            r_new = [self._spikes[i][iii] for iii in ii]
            r.append(r_new)
        return Response(a,self.stim,r)

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
        if len(self.stim)==1:
            return self._spikes[0]
        else:
            sp = [np.array([]) for i in range(len(self._spikes[0]))]
            cum_dur = 0.
            for s in range(len(self._spikes)):
                sp = [np.concatenate((sp[i],self._spikes[s][i]+cum_dur)) for i in range(len(self._spikes[0]))]
                cum_dur += self.stim[s].duration
            return sp

    def rate(self,sep=False):
        """Calculates the firing rate (in Hz) for each afferent.

        Kwargs:
            sep (bool): Whether firing rates should be separated by stimulus or
                not (default: False).

        Returns:
            Nx1 array of firing rates (NxS is sep is True).
        """
        r = np.zeros((len(self.aff),len(self.stim)))
        for i,s in enumerate(self._spikes):
            r[:,i:i+1] = (np.atleast_2d(np.array(list(map(lambda x:x.size, s)))) \
                /self.durations[i]).T
        if not sep:
            r = np.atleast_2d(np.mean(r,axis=1)).T
        return r

    def psth(self,bin=10.):
        """Calculates the time-varying response (psth) for each afferent.

        Kwargs:
            bin (float): Length of the time bins in ms (default: 10.).

        Returns:
            NxB array of firing rates (N: number of afferents, B: number of bins).
        """
        bins = np.r_[0:self.duration+bin/1000.:bin/1000.]
        return np.array(list(map(lambda x:np.histogram(x,bins=bins)[0],self.spikes)))