import pytest
import touchsim as ts

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
