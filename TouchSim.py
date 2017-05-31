import numpy as np
import MechanoTransduction
import Constants
import random
import re
from matplotlib import path
try:
    import holoviews as hv
except:
    hv = None

class Afferent(object):

    affclasses = ['SA1','RA','PC']
    affdepths = Constants.affdepths
    affparams = Constants.affparams
    affcol = Constants.affcol

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
        r = MechanoTransduction.lif_neuron(self,strain,udyn,fs)
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

    def find(self,affclass):
        return list(map(lambda x:x.affclass==affclass,self.afferents))

    def response(self,stim):
        r = list(map(lambda a:a.response(stim),self.afferents))
        return Response(self,stim,r)

    @ property
    def disp(self):
        if hv is None:
            return None
        return hv.NdOverlay({a:hv.Points(coord2plot(self.location[self.find(a),:]))\
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
        self.profile, self.profiledyn = MechanoTransduction.skin_touch_profile(
            self.trace,self.location,self.fs,self.pin_radius)

    def propagate(self,aff):
        stat_comp = MechanoTransduction.circ_load_vert_stress(
            self.profile,self.location,self.pin_radius,aff.location,aff.depth)
        dyn_comp = MechanoTransduction.circ_load_dyn_wave(
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
                [coord2plot(self.aff.location[self.aff.find(a),:]),
                r[self.aff.find(a),t:t+1]],axis=1),vdims=['Firing rate'])\
                (style=dict(color=Afferent.affcol[a])) for a in Afferent.affclasses})
        return hv.HoloMap(hm)

def affpop_single_models(**args):
    affclass = args.pop('affclass',Afferent.affparams.keys())
    a = AfferentPopulation()
    for t in affclass:
        for i in range(Afferent.affparams.get(t).shape[0]):
            a.afferents.append(Afferent(t,idx=i,**args))
    return a

def affpop_grid(**args):
    affclass = args.pop('affclass',Afferent.affclasses)
    dist = args.pop('dist',1.)
    max_extent = args.pop('max_extent',10.)
    idx = args.pop('idx',None)

    locs = np.r_[-max_extent/2:max_extent/2+dist:dist]

    a = AfferentPopulation()

    for l1 in np.nditer(locs):
        for l2 in np.nditer(locs):
            if idx is None:
                a_sub = affpop_single_models(
                    location=np.array([l1,l2]),**args)
                a.afferents.extend(a_sub.afferents)
            else:
                for t in affclass:
                    a.afferents.append(
                        Afferent(t,location=np.array([l1,l2]),idx=idx,**args))
    return a

def affpop_hand(**args):
    affclass = args.pop('affclass',Afferent.affparams.keys())
    region = args.pop('region',None)

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
            xy = np.dot(xy,Constants.rot2coord)/Constants.pxl_per_mm
            for l in range(xy.shape[0]):
                affpop.afferents.append(Afferent(a,location=xy[l,:],**args))
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

def stim_indent_shape(shape,trace,**args):
    if type(trace) is Stimulus:
        t = trace.trace[0:1]
        if 'fs' not in args:
            args['fs'] = trace.fs
        if 'pin_radius' not in args:
            args['pin_radius'] = trace.pin_radius
    else:
        t = np.reshape(np.atleast_2d(trace),(1,-1))

    if 'offset' in args:
        shape += np.tile(args.pop('offset'),shape.shape)

    return Stimulus(trace=np.tile(t,(shape.shape[0],1)),location=shape,**args)

def shape_bar(**args):
    width = args.get('width',1.)
    height = args.get('height',.5)
    angle = np.deg2rad(args.get('angle',0.))
    pins_per_mm = args.get('pins_per_mm',10)

    xy = np.mgrid[-width/2.:width/2.:width*pins_per_mm*1j,
        -height/2.:height/2.:height*pins_per_mm*1j]
    xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
    return np.dot(np.array([[np.cos(angle),-np.sin(angle)],
        [np.sin(angle),np.cos(angle)]]),xy.T).T

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

def disp_hand(region=None):

    if region is None:
        idx = range(20)
    else:
        match = re.findall('[dDpPwWmMdDfFtT]\d?',region)
        idx = [i for i,x in enumerate(Constants.regionprop_tags) if x[0]==match[0]]
        if len(match)>1:
            idx = set(idx).intersection([i for i,x in enumerate(Constants.regionprop_tags) if x[1]==match[1]])
    return hv.Path(list(map(lambda x:x.T, [Constants.regionprop_boundary[i] for i in idx])))\
        (style=dict(color='k'))

def coord2plot(locs):
    return np.dot(locs,Constants.rot2plot)*Constants.pxl_per_mm + Constants.orig
