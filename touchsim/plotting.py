import matplotlib.pyplot as plt
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
        fill (dict): Plots filled regions rather than outlines with intensity values
            provided in the dictionary for each region(s) (default: None).
        tags (bool): Adds region tags to outline (default: False).
        coord (float): if set, plot coordinate axes with specified lengths
            in mm (default: None).
        
    Kwargs for Stimulus:
        spatial = Plots pin positions spatially if true, otherwise plots pin traces
            over time (default: False).
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
        elif type(obj) is Afferent:
            return plot_afferent_population(AfferentPopulation(obj),**args)
        elif type(obj) is Stimulus:
            return plot_stimulus(obj,**args)
        elif type(obj) is Response:
            return plot_response(obj,**args)
        elif type(obj) is Surface:
            return plot_surface(obj,**args)
    raise RuntimeError("Plotting of " + str(type(obj)) + " objects not supported.")

def plot_afferent_population(obj, **args):
    size = args.get('size', 1)
    for a in Afferent.affclasses:
        locations = obj.surface.hand2pixel(obj.location[obj.find(a), :])
        plt.scatter(locations[:, 0], locations[:, 1], s=size, c=[Afferent.affcol[a]], label=a)
    plt.gca().set_aspect('equal')
    plt.legend()

def plot_stimulus(obj, **args):
    spatial = args.get('spatial', False)
    bin = args.get('bin', [0, obj.duration])

    if not spatial:
        for i, trace in enumerate(obj.trace):
            plt.plot(obj.time, trace, label=f"Pin {i}")
        plt.gca().set_xlabel("Time (s)")
        plt.gca().set_ylabel("Trace")
        plt.xlim(bin[0], bin[1])
        plt.legend()
    else:
        surface = args.get('surface', hand_surface)
        idx = np.searchsorted(obj.time, bin)
        c = np.mean(obj.trace[:, idx[0]:idx[1]],axis=1)
        c = (c - np.min(obj.trace[:, idx[0]:idx[1]])) / (np.max(obj.trace[:, idx[0]:idx[1]]) - np.min(obj.trace[:, idx[0]:idx[1]]))
        
        locs = surface.hand2pixel(obj.location)
        
        plt.gca().set_aspect('equal')
        plt.scatter(locs[:,0], locs[:,1], c=c, cmap='viridis', s=obj.pin_radius * surface.pxl_per_mm)



def plot_response(obj, **args):
    spatial = args.get('spatial', False)
    bin = args.get('bin', [0, obj.duration])
    if not spatial:
        for i, spikes in enumerate(obj.spikes):
            if len(spikes) > 0:
                plt.vlines(spikes, i, i + 0.5, color=Afferent.affcol[obj.aff.afferents[i].affclass])
        plt.gca().set_xlabel("Time (s)")
        plt.gca().set_ylabel("Neuron Index")
        plt.xlim(bin[0], bin[1])
    else:
        scale = args.get('scale', True)
        scaling_factor = args.get('scaling_factor', 2)
        
        rates = obj.psth(bins=bin)
        locs = obj.aff.surface.hand2pixel(obj.aff.location)
        for i, rate in enumerate(rates):
            plt.scatter(locs[i, 0], locs[i, 1], s=rate * scaling_factor, c=Afferent.affcol[obj.aff.afferents[i].affclass])
        plt.gca().set_aspect('equal')

def plot_surface(obj, **args):
    region = args.get('region', None)
    idx = obj.tag2idx(region)
    tags = args.get('tags', False)
    coord = args.get('coord', None)

    for i in idx:
        boundary = obj.boundary[i]
        plt.plot(boundary[:, 0], boundary[:, 1], 'k')
        
    if coord is not None:
        t1 = obj.hand2pixel([[0, 0],[0, coord]])
        t2 = obj.hand2pixel([[0, 0],[coord, 0]])
        plt.plot(t1[:,0], t1[:,1], 'r')
        plt.plot(t2[:,0], t2[:,1], 'r')
    
    if tags:
        for i in idx:
            center = obj._centers[i]
            plt.text(center[0], center[1], f"{i} {''.join(obj.tags[i])}")
    plt.gca().set_aspect('equal')
