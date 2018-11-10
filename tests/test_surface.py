import pytest
import touchsim as ts
from matplotlib import path

def test_tags2idx():
    idx = ts.hand_surface.tag2idx(None)
    for i in range(ts.hand_surface.num):
        assert i in idx

    idx = ts.hand_surface.tag2idx('D1')
    assert 15 in idx and 17 in idx

    idx = ts.hand_surface.tag2idx('D2')
    assert 2 in idx and 5 in idx and 9 in idx

    idx = ts.hand_surface.tag2idx('D2d')
    assert 2 in idx

    idx = ts.hand_surface.tag2idx('P')
    assert 12 in idx and 13 in idx and 14 in idx and\
        16 in idx and 18 in idx and 19 in idx

def test_locate():
    assert ts.hand_surface.locate([0.,0.])[0][0]=='D2d_t'

def test_sample_uniform():
    loc = ts.hand_surface.sample_uniform('D2d',num=1,seed=1)
    assert path.Path(ts.hand_surface.pixel2hand(
        ts.hand_surface.boundary[ts.hand_surface.tag2idx('D2d')[0]])
        ).contains_points(loc)[0]

    assert not path.Path(ts.hand_surface.pixel2hand(
        ts.hand_surface.boundary[ts.hand_surface.tag2idx('D1d')[0]])
        ).contains_points(loc)[0]

    loc = ts.hand_surface.sample_uniform([2],num=1,seed=1)
    assert path.Path(ts.hand_surface.pixel2hand(
        ts.hand_surface.boundary[ts.hand_surface.tag2idx('D2d')[0]])
        ).contains_points(loc)[0]

    loc = ts.hand_surface.sample_uniform(2,num=1,seed=1)
    assert path.Path(ts.hand_surface.pixel2hand(
        ts.hand_surface.boundary[ts.hand_surface.tag2idx('D2d')[0]])
        ).contains_points(loc)[0]

def test_distance():
    s = ts.stim_ramp(loc=[0.,10.])
    a = ts.Afferent('SA1',surface=ts.hand_surface)
    with pytest.warns(Warning):
        r = a.response(s)

    s = ts.stim_ramp()
    a = ts.Afferent('SA1',location=[0.,10.],surface=ts.hand_surface)
    with pytest.warns(Warning):
        r = a.response(s)

    shape = ts.shape_circle(center=[0.,6.],radius=5.,pins_per_mm=1)
    s = ts.stim_indent_shape(shape,ts.stim_ramp())
    a = ts.Afferent('SA1',surface=ts.hand_surface)
    with pytest.warns(Warning):
        r = a.response(s)
