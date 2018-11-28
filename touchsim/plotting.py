import holoviews as hv
import re
import numpy as np
import warnings
from math import ceil

from .classes import Afferent,AfferentPopulation,Stimulus,Response
from .surface import Surface,hand_surface

def plot(obj=hand_surface,**args):
    """A visual representation of an AfferentPopulation, a Stimulus, a Response,
    or a Surface object, depending on what type of object the function is called
    with.

    Args:
        obj: AfferentPopulation, Stimulus, Response, or Surface object to be
            plotted (default:hand_surface).

    Kwargs for Surface:
        region (string): Tag(s) to only plot selected regions (default: None).
        tags (bool): Adds region tags to outline (default: False).
        coord (float): if set, plot coordinate axes with specified lengths
            in mm (default: None).
        locator (bool): Dynamically shows current cursor position (only works with
            'bokeh' plotting backend; default: False).

    Kwargs for Stimulus:
        spatial = Plots pin positions spatially if true, otherwise plots pin traces
            over time (default: False).
        grid = Plots ping traces in separate panels instead of overlaid, only
            meaningful when spatial=True (default: False).
        sur = Surface to use for underlying coordinate system only
            meaningful when spatial=True (default: hand_surface).
        bin = Width of time bins in ms, used when generating animations (default: Inf).

    Kwargs for Response:
        spatial = Plots spatial response plot if true, otherwise plots spike trains
            over time (default: False).
        scale = Scales spatial response indicators by firing rate if true, otherwise
            indicates firing rate through color (default: True).
        scaling_factor = Sets response scaling factor, only meaningful, if scale=True
            (default: 2)
        bin = Width of time bins in ms, used when generating animations (default: Inf).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        sur = args.get('surface',hand_surface)
        mid = (bins[1:] + bins[:-1]) / 2.
        d = np.array([np.interp(mid,obj.time,obj.trace[i])\
            for i in range(obj.trace.shape[0])])
        d = (d-np.min(d))
        d = 1 - d/np.max(d)

        hm = dict()
        locs = sur.hand2pixel(obj.location)
        rad = obj.pin_radius*sur.pxl_per_mm
        for t in range(num):
            p = hv.Polygons([{('x','y'):hv.Ellipse(locs[l,0],locs[l,1],2*rad).array(),
                'z':d[l,t]} for l in range(obj.location.shape[0])],vdims='z').opts(
                plot=dict(color_index='z',aspect='equal'),
                style=dict(linewidth=0.,line_width=0.01)).options(cmap='fire')
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
            spikes[i] = s.opts(plot=dict(position=0.1*i,spike_length=0.1),
                style=dict(color=tuple(
                Afferent.affcol[obj.aff.afferents[idx[i]].affclass])))
        hvobj = hv.NdOverlay(spikes).opts(plot=dict(yaxis=None))

    else:
        scale = args.get('scale',True)
        scaling_factor = args.get('scaling_factor',2)
        bin = args.get('bin',float('Inf'))
        if np.isinf(bin):
            r = obj.rate()
        else:
            r = np.float64(obj.psth(bin))
        hm = dict()
        for t in range(r.shape[1]):
            points = dict()
            for a in Afferent.affclasses:
                idx = np.logical_and(obj.aff.find(a),np.logical_not(r[:,t]==0))
                if np.sum(np.nonzero(idx))==0:
                    idx[0] = True
                p = hv.Points(np.concatenate(
                    [obj.aff.surface.hand2pixel(
                    obj.aff.location[idx,:]),
                    r[idx,t:t+1]],axis=1),vdims=['Firing rate'])
                if scale:
                    points[a] = p.opts(style=dict(color=tuple(Afferent.affcol[a])),
                        plot=dict(size_index=2,scaling_factor=scaling_factor,
                        aspect='equal'))
                else:
                    points[a] = p.opts(plot=dict(color_index=2,aspect='equal')
                        ).options(cmap='fire_r')
            hm[t] = hv.NdOverlay(points)
        hvobj = hv.HoloMap(hm,kdims='Time bin [' + str(bin) + ' ms]')
    return hvobj

def plot_surface(obj,**args):
    region = args.get('region',None)
    idx = obj.tag2idx(region)
    tags = args.get('tags',False)
    coord = args.get('coord',None)
    locator = args.get('locator',False)
    filled = args.get('filled',False)

    amin = np.min(obj.bbox_min[idx],axis=0)
    amax = np.max(obj.bbox_max[idx],axis=0)
    wh = amax-amin
    if np.min(wh)<250:
        wh = wh/np.min(wh)*250
    hvobj = hv.Path([obj.boundary[i] for i in idx]).opts(
        style=dict(color='k'),plot=dict(yaxis=None,xaxis=None,aspect='equal',
        width=int(ceil(wh[0])),height=int(ceil(wh[1]))))

    if filled:
        imin = np.min(obj.bbox_min[idx],axis=0).astype(int)
        imax = np.max(obj.bbox_max[idx],axis=0).astype(int)
        im = np.nan*np.zeros(tuple((imax-imin+1)[::-1].tolist()))
        for i in idx:
            im[obj._coords[i][:,1]-imin[1],obj._coords[i][:,0]-imin[0]] = i
        hvobj = hv.Image(np.flipud(im),bounds=(imin[0],imin[1],imax[0],imax[1])).opts(
            plot=dict(yaxis=None,xaxis=None))

    if coord is not None:
        hvobj *= hv.Curve([obj.hand2pixel((0,0)),obj.hand2pixel((coord,0))]) *\
            hv.Curve([obj.hand2pixel((0,0)),obj.hand2pixel((0,coord))])
    if tags:
        hvobj *= hv.Labels({'x': [obj._centers[i][0] for i in idx],
            'y': [obj._centers[i][1] for i in idx],
            'Label': [str(i) + ' ' + ''.join(obj.tags[i]) for i in idx]})

    # show cursor position in hand coordinates (works only in bokeh)
    if locator:
        pointer = hv.streams.PointerXY(x=0,y=0)
        dm = hv.DynamicMap(lambda x, y: hvobj*hv.Text(x,y+5,
            '(%d,%d)' % tuple(obj.pixel2hand(np.array([x,y])))),streams=[pointer])
        return dm

    return hvobj

def figsave(hvobj,filename,**args):
    """Saves a plot to an image file.

    Args:
        hvobj (holoviews object): Plot to be saved.
        filename (string): Filename (without extension).

    Kwargs:
        fmt (string): Image format (default: 'png'). Use 'gif' when saving
            animations.
        size (int): Figure size in percent (default: 100).
        dpi (int): Figure resolution in dots per inch.
        fps (int): Frames per second for animations.
    """
    fmt = args.pop('fmt','png')
    hv.renderer('matplotlib').instance(**args).save(hvobj, filename, fmt=fmt)
