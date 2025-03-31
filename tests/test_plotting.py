import pytest
import touchsim as ts
from touchsim.plotting import plot


def test_plot_surface_mpl():
    plot()
    plot(labels=True,coord=10,locator=True)
    plot(region='D2d')

def test_plot_stimulus_mpl():
    s = ts.stim_ramp(len=0.25,amp=.1,ramp_len=0.05)
    plot(s)

    s += ts.stim_sine(freq=25.,len=.25,loc=[1.,1.])
    plot(s)
    plot(s,grid=True)

    with pytest.warns(Warning):
        s = ts.stim_indent_shape(ts.shape_circle(hdiff=0.5),ts.stim_ramp(len=0.1))
    plot(s,spatial=True)

def test_plot_affpop_mpl():
    a = ts.affpop_hand(region='D2d')
    plot(a)

def test_plot_response_mpl():
    a = ts.affpop_hand(region='D2')
    s = ts.stim_sine(freq=50.,amp=0.1)
    r = a.response(s)
    plot(r)
    plot(r,spatial=True)
