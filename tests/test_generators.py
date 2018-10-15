import pytest
import touchsim as ts
import numpy as np
from math import isclose

def test_affpop_default():
    a = ts.affpop_single_models()
    a = ts.affpop_linear()
    a = ts.affpop_grid()
    a = ts.affpop_hand()
    a = ts.affpop_surface()

def test_stim_default():
    s = ts.stim_sine()
    s = ts.stim_noise()
    s = ts.stim_impulse()
    s = ts.stim_ramp()

def test_affpop_hand_seed():
    a = ts.affpop_hand(seed=0)

    assert(len(a)==12535)
    assert(a[0].idx==3)

def test_affpop_single_models():
    a = ts.affpop_single_models()
    assert(len(a)==17)

    a = ts.affpop_single_models(affclass='SA1')
    assert(len(a)==4)

    a = ts.affpop_single_models(affclass='RA')
    assert(len(a)==9)

    a = ts.affpop_single_models(affclass='PC')
    assert(len(a)==4)

    a = ts.affpop_single_models(affclass=['SA1','RA'])
    assert(len(a)==13)

    with pytest.raises(Exception):
        a = ts.affpop_single_models(idx=0)

    a = ts.affpop_single_models(noisy=False)
    for n in a.noisy:
        assert not n

def test_shape_bar_hdiff():
    shape = ts.shape_bar(hdiff=0.5)

    assert shape.shape[1]==3
    assert isclose(np.min(shape[:,2]),-0.5,rel_tol=0.05)
    assert np.max(shape[:,2])==0.

def test_shape_bar_nohdiff():
    shape = ts.shape_bar()

    assert shape.shape[1]==3
    assert np.min(shape[:,2])==0.
    assert np.max(shape[:,2])==0.
