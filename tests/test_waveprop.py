import touchsim as ts

rate_slack = 1.
timing_slack = 0.00025

def test_PC_waveprop1():
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

def test_PC_waveprop2():
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
