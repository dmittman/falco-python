"""FALCO regression test of WFSC with a filtered Lyot coronagraph."""
from copy import deepcopy
import numpy as np
import os
from math import isclose

import falco
import falco.hdf5utils as hdf5utils

import config_wfsc_flc as CONFIG


def test_wfsc_flc():

    mp = deepcopy(CONFIG.mp)
    mp.flagPlot = False
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

    # Generate the label associated with this trial
    mp.runLabel = 'testing_wfsc_flc'

    # Perform the Wavefront Sensing and Control
    out = falco.setup.flesh_out_workspace(mp)
    falco.wfsc.loop(mp, out)

    fn_hdf5 = os.path.join(mp.path.brief, f"{mp.runLabel}_snippet.h5")
    out_from_hdf5 = hdf5utils.load_object_from_hdf5(fn_hdf5)

    print(out.IrawCorrHist[-1])
    print(out.IestScoreHist[-1])
    print(out.IincoCorrHist[-1])
    print(out.complexProjection[1, 0])
    print(out.dm1.Spv[-1])
    print(out.thput[-1])
    print(out.log10regHist)

    # Tests:
    Iend = out.IrawCorrHist[-1]  # 1.1157e-5 in matlab, 1.073e-05 in python
    assert isclose(Iend, 8.12e-6, abs_tol=1e-6)

    Iest = out.IestScoreHist[-1]  # 8.15e-06 in matlab, 8.123e-6 in python
    assert isclose(Iest, 1.53e-5, abs_tol=3e-7)

    # Iinco = out.IincoCorrHist[-1]  # 1.28e-5 in matlab, 1.165e-5 in python
    # assert isclose(Iinco, 1.165e-5, abs_tol=3e-7)

    complexProj = out.complexProjection[1, 0]  # 0.74 in matlab, 0.82 in python
    assert isclose(complexProj, 0.88, abs_tol=0.02)

    dm1pv = out.dm1.Spv[-1]  # 5.6956e-08 in matlab, 5.75e-8 in python
    assert isclose(dm1pv, 6.09e-8, abs_tol=1e-9)

    thput = out.thput[-1]  # 0.1493 in matlab, 0.1492 in python
    assert isclose(thput, 0.1486, abs_tol=1e-3)

    assert np.allclose(out.log10regHist, np.array([-2, -2, -2]), rtol=1e-2)
    assert np.allclose(out_from_hdf5.log10regHist, out.log10regHist, rtol=1e-12)
    assert isclose(out_from_hdf5.IrawCorrHist[-1], out.IrawCorrHist[-1], abs_tol=1e-14)


if __name__ == '__main__':
    test_wfsc_flc()
