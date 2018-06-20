import holoviews as hv
import re
import numpy as np

from .classes import Afferent,AfferentPopulation,Stimulus,Response
from .surface import Surface,hand_surface

def plot(obj=hand_surface,**args):
    if type(obj) is AfferentPopulation:
        return plot_afferent_population(obj,**args)
    elif type(obj) is Stimulus:
        return plot_stimulus(obj,**args)
    elif type(obj) is Response:
        return plot_response(obj,**args)
    elif type(obj) is Surface:
        return plot_surface(obj,**args)
    raise RuntimeError("Plotting of " + str(type(obj)) + " objects not supported.")

def plot_afferent_population(obj,**args):
    points = dict()
    for a in Afferent.affclasses:
        p = hv.Points(
            obj.surface.hand2pixel(obj.location[obj.find(a),:]))
        points[a] = p.opts(plot=dict(aspect='equal'),
            style=dict(color=tuple(Afferent.affcol[a])))
    return hv.NdOverlay(points)

def plot_stimulus(obj,**args):
    spatial = args.get('spatial',False)
    bin = args.get('bin',float('Inf'))
    if np.isinf(bin):
        bins = np.array([0,obj.duration])
        num = 1
    else:
        bins = np.r_[0:obj.duration+bin/1000.:bin/1000.]
        num = bins.size-1

    if not spatial:
        grid = args.get('grid',False)
        hm = dict()
        tmin = np.min(obj.trace)
        tmax = np.max(obj.trace)
        for t in range(num):
            if num==1:
                d = {i:hv.Curve((obj.time,obj.trace[i]))\
                    for i in range(obj.trace.shape[0])}
            else:
                d = {i:hv.Curve((obj.time,obj.trace[i]))*\
                    hv.Curve([(bins[t+1],tmin),(bins[t+1],tmax)])
                    for i in range(obj.trace.shape[0])}
            if grid:
                hvobj = hv.NdLayout(d)
            else:
                hvobj = hv.NdOverlay(d)
            hm[t] = hvobj
        hvobj = hv.HoloMap(hm,kdims='Time bin [' + str(bin) + ' ms]').collate()
    else:
        mid = (bins[1:] + bins[:-1]) / 2.
        d = np.array([np.interp(mid,obj.time,obj.trace[i])\
            for i in range(obj.trace.shape[0])])
        d = (d-np.min(d))
        d = 1 - d/np.max(d)

        hm = dict()
        for t in range(num):
            p = hv.Points(np.hstack((np.atleast_2d(hand_surface.hand2pixel(\
                obj.location)),d[:,t:t+1])),vdims=['Depth'])
            p = p.opts(plot=dict(color_index=2,scaling_factor=4,aspect='equal'))
            hm[t] = p
        hvobj = hv.HoloMap(hm,kdims='Time bin [' + str(bin) + ' ms]')
    return hvobj

def plot_response(obj,**args):
    spatial = args.get('spatial',False)
    if not spatial:
        idx = [i for i in range(len(obj.spikes)) if len(obj.spikes[i]>0)]
        spikes = dict()
        for i in range(len(idx)):
            s = hv.Spikes(obj.spikes[idx[i]], kdims=['Time'])
            spikes[i] = s.opts(plot=dict(position=0.1*i),
                style=dict(color=tuple(
                Afferent.affcol[obj.aff.afferents[idx[i]].affclass])))
        hvobj = hv.NdOverlay(spikes).opts(plot=dict(yaxis=None))

    else:
        bin = args.get('bin',float('Inf'))
        if np.isinf(bin):
            r = obj.rate()
        else:
            r = obj.psth(bin)
        hm = dict()
        for t in range(r.shape[1]):
            points = dict()
            for a in Afferent.affclasses:
                p = hv.Points(np.concatenate(
                    [obj.aff.surface.hand2pixel(
                    obj.aff.location[obj.aff.find(a),:]),
                    r[obj.aff.find(a),t:t+1]],axis=1),vdims=['Firing rate'])
                points[a] = p.opts(style=dict(color=tuple(Afferent.affcol[a])),
                    plot=dict(size_index=2,scaling_factor=2,aspect='equal'))
            hm[t] = hv.NdOverlay(points)
        hvobj = hv.HoloMap(hm,kdims='Time bin [' + str(bin) + ' ms]')
    return hvobj

def plot_surface(obj,**args):
    region = args.get('region',None)
    idx = obj.tag2idx(region)
    labels = args.get('labels',False)
    coord = args.get('coord',None)
    hvobj = hv.Path([obj.boundary[i] for i in idx]).opts(
        style=dict(color='k'),plot=dict(yaxis=None,xaxis=None,aspect='equal'))
    if coord is not None:
        hvobj *= hv.Curve([obj.hand2pixel((0,0)),obj.hand2pixel((coord,0))]) *\
            hv.Curve([obj.hand2pixel((0,0)),obj.hand2pixel((0,coord))])
    if labels:
        hvobj *= hv.Labels({'x': [obj.centers[i][0] for i in idx],
            'y': [obj.centers[i][1] for i in idx],
            'Label': [str(idx[i]) + ' ' + ''.join(obj.tags[i]) for i in idx]})
    return hvobj

def figsave(hvobj,filename,**args):
    if type(hvobj) is hv.core.spaces.HoloMap:
        hv.renderer('matplotlib').instance(**args).save(hvobj, filename, fmt='gif')
    else:
        hv.renderer('matplotlib').instance(**args).save(hvobj, filename, fmt='png')
