import holoviews as hv
import re
import numpy as np

from .classes import Afferent,AfferentPopulation,Stimulus,Response
from .surface import Surface,hand_surface

def plot(obj=hand_surface,**args):
    if type(obj) is AfferentPopulation:
        hvobj = hv.NdOverlay({a:hv.Points(obj.surface.hand2pixel(obj.location[obj.find(a),:]))\
            (style=dict(color=Afferent.affcol[a])) for a in Afferent.affclasses})

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
            hvobj = hv.NdOverlay({i: hv.Spikes(obj.spikes[idx[i]], kdims=['Time'])\
                (plot=dict(position=0.1*i),
                style=dict(color=Afferent.affcol[obj.aff.afferents[idx[i]].affclass]))\
                for i in range(len(idx))})(plot=dict(yaxis='bare'))

        else:
            bin = args.get('bin',float('Inf'))
            if np.isinf(bin):
                r = obj.rate()
            else:
                r = obj.psth(bin)
            hm = dict()
            for t in range(r.shape[1]):
                hm[t] = hv.NdOverlay({a:hv.Points(np.concatenate(
                    [obj.aff.surface.hand2pixel(obj.aff.location[obj.aff.find(a),:]),
                    r[obj.aff.find(a),t:t+1]],axis=1),vdims=['Firing rate'])\
                    (style=dict(color=Afferent.affcol[a])) for a in Afferent.affclasses})
            hvobj = hv.HoloMap(hm)

    elif type(obj) is Surface:
        region = args.get('region',None)
        idx = obj.tag2idx(region)
        return hv.Path([obj.boundary[i] for i in idx])(style=dict(color='k'))

    return hvobj
