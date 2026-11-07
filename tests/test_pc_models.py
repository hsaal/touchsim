import numpy as np
import touchsim as ts


def _rates(idx):
    s = ts.stim_sine(freq=np.array([20., 300.]), amp=np.array([0.05, 0.005]), len=0.5)
    return [ts.Afferent('PC', idx=k, noisy=False).response(s).rate()[0, 0] for k in idx]


def test_stock_is_default():
    a = ts.Afferent('PC', idx=0)
    assert a.stages is None
    assert np.allclose(a.parameters[:12], ts.constants.affparamsPC_stock[0, :12])


def test_refit_models():
    stock = _rates(range(4))
    ts.set_pc_models('refit')
    try:
        assert ts.Afferent('PC', idx=0).stages['dz'] is not None
        assert ts.Afferent('PC', idx=3).stages['dz'] is not None
        assert ts.Afferent('PC', idx=2).stages is None
        assert np.allclose(ts.Afferent('PC', idx=1).parameters[:12],
                           ts.constants.affparamsPC_refit[1, :12])
        refit = _rates(range(4))
        assert refit != stock
        # without its dead zone, refit PC model 0 is the plain IF stage
        a = ts.Afferent('PC', idx=0, noisy=False)
        s = ts.stim_sine(freq=np.array([20.]), amp=np.array([0.05]), len=0.5)
        r_dz = a.response(s).rate()[0, 0]
        a.stages = {'dz': (0., 0.07)}
        assert a.response(s).rate()[0, 0] >= r_dz
    finally:
        ts.set_pc_models('stock')
    assert _rates(range(4)) == stock
