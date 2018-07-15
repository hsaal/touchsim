import pytest
import touchsim as ts
import holoviews as hv
from touchsim.plotting import plot

def test_plot_surface_mpl():
    renderer = hv.renderer('matplotlib')
    obj = plot()
    obj = plot(labels=True,coord=10,locator=True)
    obj = plot(region='D2d')

def test_plot_stimulus_mpl():
    renderer = hv.renderer('matplotlib')
    s = ts.stim_ramp(len=0.25,amp=.1,ramp_len=0.05)
    obj = plot(s)

    s += ts.stim_sine(freq=25.,len=.25,loc=[1.,1.])
    obj = plot(s)
    obj = plot(s,grid=True)

    with pytest.warns(Warning):
        s = ts.stim_indent_shape(ts.shape_circle(hdiff=0.5),ts.stim_ramp(len=0.1))
    obj = plot(s,spatial=True)

def test_plot_affpop_mpl():
    renderer = hv.renderer('matplotlib')
    a = ts.affpop_hand(region='D2d')
    obj = plot(a)
    obj = plot(a)['PC']
    obj = plot(a)[:,120:140,450:475]

def test_plot_response_mpl():
    renderer = hv.renderer('matplotlib')
    a = ts.affpop_hand(region='D2')
    s = ts.stim_sine(freq=50.,amp=0.1)
    r = a.response(s)
    obj = plot(r)
    obj = plot(r)[:,0:0.2]
    obj = plot(r,spatial=True)
    obj = plot(r,spatial=True,scale=False)[:,'RA']

def plot_overlay_mpl():
    renderer = hv.renderer('matplotlib')
    a = ts.affpop_hand(region='D2d')
    obj = plot(region='D2d') * plot(a)['PC']
    obj = (plot(region='D2d') * plot(a)['SA1']) +\
        (plot(region='D2d') * plot(a)['RA']) +\
        (plot(region='D2d') * plot(a)['PC'])
