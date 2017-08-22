import holoviews as hv
import re
import numpy as np

from .classes import Afferent,AfferentPopulation,Stimulus,Response
from .surface import Surface,hand_surface

def plot(obj=hand_surface,**args):
    if type(obj) is AfferentPopulation:
        points = dict()
        for a in Afferent.affclasses:
            p = hv.Points(
                obj.surface.hand2pixel(obj.location[obj.find(a),:]))
            points[a] = p.opts(style=dict(color=tuple(Afferent.affcol[a])))
        hvobj = hv.NdOverlay(points)

    elif type(obj) is Stimulus:
        grid = args.get('grid',False)
        d = {i:hv.Curve((obj.time,obj.trace[i]))
             for i in range(obj.trace.shape[0])}
        if grid:
            hvobj = hv.NdLayout(d)
        else:
            hvobj = hv.NdOverlay(d)

    elif type(obj) is Response:
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
                        plot=dict(size_index=2,scaling_factor=2))
                hm[t] = hv.NdOverlay(points)
            hvobj = hv.HoloMap(hm)

    elif type(obj) is Surface:
        region = args.get('region',None)
        idx = obj.tag2idx(region)
        return hv.Path([obj.boundary[i] for i in idx]).opts(
            style=dict(color='k'),plot=dict(yaxis=None,xaxis=None))

    return hvobj
