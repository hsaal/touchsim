import touchsim as ts
import numpy as np

delay_slack = 2

def test_profile_1pin_ramp():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    '''
    s = ts.stim_ramp(pin_radius=1.)

    assert np.around(max(s._profile.flatten()),decimals=3) == 0.119

def test_profiledyn_1pin_ramp():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    '''
    s = ts.stim_ramp(pin_radius=1.)

    assert np.around(max(s._profiledyn.flatten()),decimals=3) == 2.391
    assert np.around(min(s._profiledyn.flatten()),decimals=3) == -2.391

def test_profile_1pin_sine():
    '''
    Matlab code:
    s = stim_sine(10,0.25,0);
    '''
    s = ts.stim_sine(freq=10,amp=0.25)

    assert np.around(max(s._profile.flatten()),decimals=4) == 0.0149
    assert np.around(min(s._profile.flatten()),decimals=4) == -0.0149

def test_profiledyn_1pin_sine():
    '''
    Matlab code:
    s = stim_sine(10,0.25,0);
    '''
    s = ts.stim_sine(freq=10,amp=0.25)

    assert np.around(max(s._profiledyn.flatten()),decimals=3) == 0.935
    assert np.around(min(s._profiledyn.flatten()),decimals=3) == -0.935

def test_static_1pin_ramp():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    a = Afferent('SA1');
    [~,stat_comp] = s.propagate(a,false);
    '''
    s = ts.stim_ramp(pin_radius=1.)
    a = ts.Afferent('SA1')
    stat_comp,_,_ = s.propagate(a)

    assert np.around(max(stat_comp)[0],decimals=4) == 0.0203

def test_static_1pin_sine():
    '''
    Matlab code:
    s = stim_sine(10,0.25,0);
    a = Afferent('SA1');
    [~,stat_comp] = s.propagate(a,false);
    '''
    s = ts.stim_sine(freq=10,amp=0.25)
    a = ts.Afferent('SA1')
    stat_comp,_,_ = s.propagate(a)

    assert np.around(max(stat_comp)[0],decimals=4) == 0.0107

def test_static_2pin_sync():
    '''
    Matlab code:
    s1 = stim_sine(10,0.5,0,[],[],[],[],0.5);
    s2 = stim_sine(10,0.5,0,[],[1 1],[],[],0.5);
    s = Stimulus([s1.trace s2.trace],[s1.location;s2.location],5000,0.5);
    a = Afferent('SA1','location',[.5 .5]);
    [~,stat_comp] = s.propagate(a,false);
    '''
    s = ts.stim_sine(freq=10,amp=0.5,pin_radius=0.5)
    s += ts.stim_sine(freq=10,amp=0.5,pin_radius=0.5,loc=[1.,1.])
    a = ts.Afferent('SA1',location=[.5,.5])
    stat_comp,_,_ = s.propagate(a)

    assert np.around(max(stat_comp)[0],decimals=4) == 0.0086

def test_static_2pin_async():
    '''
    Matlab code:
    s1 = stim_sine(10,0.5,90,[],[],[],[],0.5);
    s2 = stim_sine(10,0.5,0,[],[1 1],[],[],0.5);
    s = Stimulus([s1.trace s2.trace],[s1.location;s2.location],5000,0.5);
    a = Afferent('SA1','location',[.5 .5]);
    [~,stat_comp] = s.propagate(a,false);
    '''
    s = ts.stim_sine(freq=10,amp=0.5,phase=90.,pin_radius=0.5)
    s += ts.stim_sine(freq=10,amp=0.5,pin_radius=0.5,loc=[1.,1.])
    a = ts.Afferent('SA1',location=[.5,.5])
    stat_comp,_,_ = s.propagate(a)

    assert np.around(max(stat_comp)[0],decimals=4) == 0.0061

def test_dynamic_1pin_ramp():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    a = Afferent('SA1');
    [~,~,dyn_comp] = s.propagate(a,false);
    '''
    s = ts.stim_ramp(pin_radius=1.)
    a = ts.Afferent('SA1')
    _,dyn_comp,_ = s.propagate(a)

    assert np.around(max(dyn_comp)[0],decimals=3) == 13.281
    assert np.around(min(dyn_comp)[0],decimals=3) == -13.281

def test_dynamic_1pin_sine():
    '''
    Matlab code:
    s = stim_sine(10,0.25,0);
    a = Afferent('SA1');
    [~,~,dyn_comp] = s.propagate(a,false);
    '''
    s = ts.stim_sine(freq=10,amp=0.25)
    a = ts.Afferent('SA1')
    _,dyn_comp,_ = s.propagate(a)

    assert np.around(max(dyn_comp)[0],decimals=3) == 10.391
    assert np.around(min(dyn_comp)[0],decimals=3) == -10.391

def test_dynamic_2pin_sync():
    '''
    Matlab code:
    s1 = stim_sine(10,0.5,0,[],[],[],[],0.5);
    s2 = stim_sine(10,0.5,0,[],[1 1],[],[],0.5);
    s = Stimulus([s1.trace s2.trace],[s1.location;s2.location],5000,0.5);
    a = Afferent('SA1','location',[.5 .5]);
    [~,~,dyn_comp] = s.propagate(a,false);
    '''
    s = ts.stim_sine(freq=10,amp=0.5,pin_radius=0.5)
    s += ts.stim_sine(freq=10,amp=0.5,pin_radius=0.5,loc=[1.,1.])
    a = ts.Afferent('SA1',location=[.5,.5])
    _,dyn_comp,_ = s.propagate(a)

    assert np.around(max(dyn_comp)[0],decimals=3) == 16.895
    assert np.around(min(dyn_comp)[0],decimals=3) == -16.895

def test_dynamic_2pin_async():
    '''
    Matlab code:
    s1 = stim_sine(10,0.5,90,[],[],[],[],0.5);
    s2 = stim_sine(10,0.5,0,[],[1 1],[],[],0.5);
    s = Stimulus([s1.trace s2.trace],[s1.location;s2.location],5000,0.5);
    a = Afferent('SA1','location',[.5 .5]);
    [~,~,dyncomp] = s.propagate(a,false);
    '''
    s = ts.stim_sine(freq=10,amp=0.5,phase=90.,pin_radius=0.5)
    s += ts.stim_sine(freq=10,amp=0.5,pin_radius=0.5,loc=[1.,1.])
    a = ts.Afferent('SA1',location=[.5,.5])
    _,dyn_comp,_ = s.propagate(a)

    assert np.around(max(dyn_comp)[0],decimals=3) == 19.085
    assert np.around(min(dyn_comp)[0],decimals=3) == -19.085

def test_waveprop_1pin_ramp1():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    a = Afferent('SA1','location',[1 1]);
    [~,~,dyn_comp] = s.propagate(a,false);
    '''
    s = ts.stim_ramp(pin_radius=1.)
    a = ts.Afferent('SA1',location=[1.,1.])
    _,dyn_comp,_ = s.propagate(a)

    assert np.around(max(dyn_comp)[0],decimals=3) == 6.640
    assert np.around(min(dyn_comp)[0],decimals=3) == -6.640

def test_waveprop_1pin_ramp2():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],[],1);
    a = Afferent('SA1','location',[50 0]);
    [~,~,dyn_comp] = s.propagate(a,false);
    '''
    s = ts.stim_ramp(pin_radius=1.)
    a = ts.Afferent('SA1',location=[50.,0.])
    _,dyn_comp,_ = s.propagate(a)

    assert np.around(max(dyn_comp)[0],decimals=3) == 0.169
    assert np.around(min(dyn_comp)[0],decimals=3) == -0.169

def test_waveprop_1pin_sine():
    '''
    Matlab code:
    s = stim_sine(10,0.25,0);
    a = Afferent('SA1','location',[1 1]);
    [~,~,dyn_comp] = s.propagate(a,false);
    '''
    s = ts.stim_sine(freq=10,amp=0.25)
    a = ts.Afferent('SA1',location=[1.,1.])
    _,dyn_comp,_ = s.propagate(a)

    assert np.around(max(dyn_comp)[0],decimals=3) == 2.390
    assert np.around(min(dyn_comp)[0],decimals=3) == -2.390

def test_waveprop_delay_1pin1():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],'sine',1);
    a = Afferent('SA1','location',[50 0]);
    [~,~,dyn_comp] = s.propagate(a,false);
    '''
    s = ts.stim_ramp(pin_radius=1.,ramp_type='sine')
    a = ts.Afferent('SA1',location=[50.,0.])
    _,dyn_comp,_ = s.propagate(a)

    assert np.argmax(dyn_comp.flatten())>=155-delay_slack
    assert np.argmax(dyn_comp.flatten())<=155+delay_slack

def test_waveprop_delay_1pin2():
    '''
    Matlab code:
    s = stim_ramp([],[],[],[],[],'sine',1);
    a = Afferent('SA1','location',[150 0]);
    [~,~,dyn_comp] = s.propagate(a,false);
    '''
    s = ts.stim_ramp(pin_radius=1.,ramp_type='sine')
    a = ts.Afferent('SA1',location=[150.,0.])
    _,dyn_comp,_ = s.propagate(a)

    assert np.argmax(dyn_comp.flatten())>=219-delay_slack
    assert np.argmax(dyn_comp.flatten())<=219+delay_slack
