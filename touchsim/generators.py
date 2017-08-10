import numpy as np

from .classes import Afferent,AfferentPopulation,Stimulus
from .surface import Surface, hand_surface
from . import constants

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
    affclass = args.pop('affclass',Afferent.affclasses)
    if type(affclass) is not list:
        affclass = [affclass]
    region = args.pop('region',None)

    idx = hand_surface.tag2idx(region)

    affpop = AfferentPopulation(surface=hand_surface)
    for a in affclass:
        for i in idx:
            dens = np.sqrt(constants.density[(a,
                hand_surface.tags[i][2])])/10./hand_surface.pxl_per_mm

            xy = hand_surface.sample_uniform(i,dens)
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
    pin_radius = args.get('pin_radius',.5)
    pre_indent = args.get('pre_indent',0.)
    pad_len = args.get('pad_len',0.)

    trace = np.zeros(int(fs*len))
    for f,a,p in zip(np.nditer(freq),np.nditer(amp),np.nditer(phase)):
        trace += a*np.sin(p*np.pi/180. \
            + np.linspace(0.,2.*np.pi*f*len,int(fs*len)))

    apply_ramp(trace,len=ramp_len,fs=fs)
    apply_pad(trace,len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,pin_radius=pin_radius)

def stim_ramp(**args):
    amp = args.get('amp',1.)
    ramp_type = args.get('ramp_type','lin')
    len = args.get('len',1.)
    loc = np.array(args.get('loc',np.array([0.,0.])))
    fs = args.get('fs',5000.)
    ramp_len = args.get('ramp_len',.05)
    pin_radius = args.get('pin_radius',.5)
    pre_indent = args.get('pre_indent',0.)
    pad_len = args.get('pad_len',0.)

    trace = amp*np.ones(int(fs*len))
    apply_ramp(trace,len=ramp_len,fs=fs,ramp_type=ramp_type)
    apply_pad(trace,len=pad_len,fs=fs)
    trace += pre_indent

    return Stimulus(trace=trace,location=loc,fs=fs,pin_radius=pin_radius)

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
        trace[:len] *= np.cos(np.linspace(np.pi,2.*np.pi,len))/2.+.5
        trace[-len:] *= np.cos(np.linspace(0.,np.pi,len))/2.+.5
    else:
        raise RuntimeError("ramp_type must be 'lin' or 'sin'")

def apply_pad(trace,**args):
    len = args.get('len',.05)
    fs = args.get('fs',None)
    if max(len,0.)==0.:
        return
    if fs is not None:
        len = round(len*fs)
    trace = np.concatenate((np.zeros(len),trace,np.zeros(len)))
