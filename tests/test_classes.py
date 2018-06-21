import pytest
import touchsim as ts
import numpy as np

rate_slack = 1.
timing_slack = 0.00025

def test_response_multiple1():
    s = [ts.stim_sine(freq=250.,amp=0.005,fs=1000.)]
    a = ts.Afferent('PC',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=246.-rate_slack
    assert r.rate()[0,0]<=246.+rate_slack
    assert r.spikes[0][0]>=0.0042-timing_slack
    assert r.spikes[0][0]<=0.0042+timing_slack

def test_response_multiple2():
    s = [ts.stim_sine(freq=250.,amp=0.005,fs=1000.),
        ts.stim_sine(freq=250.,amp=0.005,fs=1000.)]
    a = ts.Afferent('PC',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=246.-rate_slack
    assert r.rate()[0,0]<=246.+rate_slack
    assert r.spikes[0][0]>=0.0042-timing_slack
    assert r.spikes[0][0]<=0.0042+timing_slack

def test_response_multiple3():
    s = [ts.stim_ramp(pin_radius=1.),
        ts.stim_sine(freq=250.,amp=0.005,fs=1000.)]
    a = ts.Afferent('PC',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=124.5-rate_slack
    assert r.rate()[0,0]<=124.5+rate_slack
    assert r.rate(sep=True)[0,0]>=3-rate_slack
    assert r.rate(sep=True)[0,0]<=3+rate_slack
    assert r.rate(sep=True)[0,1]>=246-rate_slack
    assert r.rate(sep=True)[0,1]<=246+rate_slack
    assert r.spikes[0][0]>=0.0004-timing_slack
    assert r.spikes[0][0]<=0.0004+timing_slack
    assert r._spikes[1][0][0]>=0.0042-timing_slack
    assert r._spikes[1][0][0]<=0.0042+timing_slack

def test_afferent_add():
    a = ts.Afferent('SA1')
    a2 = ts.Afferent('RA')
    ap = ts.AfferentPopulation(ts.Afferent('RA'),ts.Afferent('PC'))

    a_new = a + a2

    assert len(a_new)==2
    assert len(a)==1
    assert len(a2)==1

    a_new = a + ap

    assert len(a_new)==3
    assert len(a)==1
    assert len(ap)==2

def test_affpop_index():
    a = ts.affpop_single_models()

    assert len(a[0])==1
    assert len(a[0:1])==1
    assert len(a[:])==17
    assert len(a[[0,1]])==2
    assert len(a[np.array([0,1])])==2
    assert len(a[[True,False,True]])==2
    assert len(a['SA1'])==4
    assert len(a['RA'])==9
    assert len(a['PC'])==4

def test_affpop_add():
    a = ts.AfferentPopulation(ts.Afferent('SA1'))
    a2 = ts.Afferent('RA')
    ap = ts.AfferentPopulation(ts.Afferent('RA'),ts.Afferent('PC'))

    a_new = a + a2

    assert len(a_new)==2
    assert len(a)==1
    assert len(a2)==1

    a_new = a + ap

    assert len(a_new)==3
    assert len(a)==1
    assert len(ap)==2

def test_affpop_iadd():
    a = ts.AfferentPopulation(ts.Afferent('SA1'))
    a2 = ts.Afferent('RA')
    ap = ts.AfferentPopulation(ts.Afferent('RA'),ts.Afferent('PC'))

    a += a2

    assert len(a)==2
    assert len(a2)==1

    a += ap

    assert len(a)==4
    assert len(ap)==2

def test_stimulus_iadd():
    s = ts.stim_ramp(loc=[0.,0.])
    s2 = ts.stim_ramp(loc=[5.,0.])
    assert len(s)==1

    s += s2

    s3 = ts.stim_ramp(loc=[10.,0.],pin_radius=2.)
    with pytest.warns(Warning):
        s += s3
    assert len(s)==3
    assert s.pin_radius==0.5
    assert s3.pin_radius==2.

    s4 = ts.stim_ramp(loc=[0.,0.])
    with pytest.raises(Exception):
        s += s4

    s5 = ts.stim_ramp(loc=[0.,0.],fs=500.)
    with pytest.raises(Exception):
        s += s5
