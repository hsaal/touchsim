import touchsim as ts

rate_slack = 1.
timing_slack = 0.00025

def test_SA_ramp():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    a = Afferent('SA1','idx',1,'noisy',false);
    '''
    s = ts.stim_ramp(pin_radius=1.)
    a = ts.Afferent('SA1',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=22.-rate_slack
    assert r.rate()[0,0]<=22.+rate_slack
    assert r.spikes[0][0]>=0.0062-timing_slack
    assert r.spikes[0][0]<=0.0062+timing_slack

def test_SA_delay():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    a = Afferent('SA1','idx',1,'noisy',false,'delay',true);
    '''
    s = ts.stim_ramp(pin_radius=1.)
    a = ts.Afferent('SA1',idx=0,noisy=False,delay=True)
    r = a.response(s)

    assert r.rate()[0,0]>=22.-rate_slack
    assert r.rate()[0,0]<=22.+rate_slack
    assert r.spikes[0][0]>=0.0111-timing_slack
    assert r.spikes[0][0]<=0.0111+timing_slack

def test_SA_sine():
    '''
    Matlab code:
    s = stim_sine(10,0.5,0);
    a = Afferent('SA1','idx',1,'noisy',false);
    '''
    s = ts.stim_sine(freq=10.,amp=0.5)
    a = ts.Afferent('SA1',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=43.-rate_slack
    assert r.rate()[0,0]<=43.+rate_slack
    assert r.spikes[0][0]>=0.013-timing_slack
    assert r.spikes[0][0]<=0.013+timing_slack

def test_RA_ramp():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    a = Afferent('RA','idx',1,'noisy',false);
    '''
    s = ts.stim_ramp(pin_radius=1.)
    a = ts.Afferent('RA',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=7.-rate_slack
    assert r.rate()[0,0]<=7.+rate_slack
    assert r.spikes[0][0]>=0.0092-timing_slack
    assert r.spikes[0][0]<=0.0092+timing_slack

def test_RA_sine():
    '''
    Matlab code:
    s = stim_sine(25,0.5,0);
    a = Afferent('RA','idx',1,'noisy',false);
    '''
    s = ts.stim_sine(freq=25.,amp=0.5)
    a = ts.Afferent('RA',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=119.-rate_slack
    assert r.rate()[0,0]<=119.+rate_slack
    assert r.spikes[0][0]>=0.0126-timing_slack
    assert r.spikes[0][0]<=0.0126+timing_slack

def test_PC_ramp():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    a = Afferent('PC','idx',1,'noisy',false);
    '''
    s = ts.stim_ramp(pin_radius=1.)
    a = ts.Afferent('PC',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=3.-rate_slack
    assert r.rate()[0,0]<=3.+rate_slack
    assert r.spikes[0][0]>=0.0004-timing_slack
    assert r.spikes[0][0]<=0.0004+timing_slack

def test_PC_sine1():
    '''
    Matlab code:
    s = stim_sine(250,0.005,0);
    a = Afferent('PC','idx',1,'noisy',false);
    '''
    s = ts.stim_sine(freq=250.,amp=0.005)
    a = ts.Afferent('PC',idx=0,noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=247.-rate_slack
    assert r.rate()[0,0]<=247.+rate_slack
    assert r.spikes[0][0]>=0.0042-timing_slack
    assert r.spikes[0][0]<=0.0042+timing_slack

def test_PC_sine2():
    '''
    Matlab code:
    s = stim_sine(250,0.01,0);
    a = Afferent('PC','idx',1,'location',[25 0],'noisy',false);
    '''
    s = ts.stim_sine(freq=250.,amp=0.01)
    a = ts.Afferent('PC',idx=0,location=[25.,0.],noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=40.-rate_slack
    assert r.rate()[0,0]<=40.+rate_slack
    assert r.spikes[0][0]>=0.0238-timing_slack
    assert r.spikes[0][0]<=0.0238+timing_slack

def test_PC_sine3():
    '''
    Matlab code:
    s = stim_sine(250,0.01,0);
    a = Afferent('PC','idx',1,'location',[50 0],'noisy',false);
    '''
    s = ts.stim_sine(freq=250.,amp=0.01)
    a = ts.Afferent('PC',idx=0,location=[50.,0.],noisy=False)
    r = a.response(s)

    assert r.rate()[0,0]>=29.-rate_slack
    assert r.rate()[0,0]<=29.+rate_slack
    assert r.spikes[0][0]>=0.043-timing_slack
    assert r.spikes[0][0]<=0.043+timing_slack
