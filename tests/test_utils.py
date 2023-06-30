import numpy as np
import torch

import foureg.utils as utils

np.random.seed(108)


def _wrapFilter(src, vecs, *args):
    dest = src.copy()
    for vec in vecs:
        _addFreq(dest, vec)

    filtered = utils.imfilter(dest, *args)
    mold, mnew = [_arrdiff(src, arr)[0] for arr in (dest, filtered)]

    assert mold * 1e-10 > mnew


def _addFreq(src, vec):
    dom = np.zeros(src.shape)
    dom += np.arange(src.shape[0])[:, np.newaxis] * np.pi * vec[0]
    dom += np.arange(src.shape[1])[np.newaxis, :] * np.pi * vec[1]

    src += np.sin(dom)

    return src


def _arrdiff(a, b):
    adiff = np.abs(a - b)
    ret = adiff.mean(), adiff.max()
    return ret


def test_subarray():
    arr = torch.arange(20)
    arr = arr.reshape((4, 5))

    # trivial subarray
    suba = utils._get_subarr(arr, (1, 1), 1).cpu().numpy()
    ret = arr[:3, :3]
    assert np.allclose(suba, ret.numpy())

    # subarray with zero radius
    suba = utils._get_subarr(arr, (1, 1), 0).cpu().numpy()
    ret = arr[1, 1]
    assert np.allclose(suba, ret.numpy())

    # subarray that wraps through two edges
    suba = utils._get_subarr(arr, (0, 0), 1).cpu().numpy()
    ret = np.zeros((3, 3), int)
    ret[1:, 1:] = arr[:2, :2]
    ret[0, 0] = arr[-1, -1]
    ret[0, 1] = arr[-1, 0]
    ret[0, 2] = arr[-1, 1]
    ret[1, 0] = arr[0, -1]
    ret[2, 0] = arr[1, -1]
    assert np.allclose(suba, ret)


def test_filter():
    src = np.zeros((20, 30))

    _wrapFilter(src, [(0.8, 0.8)], (0.8, 1.0))
    _wrapFilter(src, [(0.1, 0.2)], None, (0.3, 0.4))

    src2 = _addFreq(src.copy(), (0.1, 0.4))
    _wrapFilter(src2, [(0.8, 0.8), (0.1, 0.2)], (0.8, 1.0), (0.3, 0.4))


def test_Argmax_ext():
    src = np.array([[1, 3, 1], [0, 0, 0], [1, 3.01, 0]])
    infres = utils._argmax_ext(src, "inf")  # element 3.01
    assert tuple(infres) == (2.0, 1.0)

    n10res = utils._argmax_ext(src, 10)  # element 1 in the rows with 3s
    n10res = np.round(n10res)
    assert tuple(n10res) == (1, 1)


def test_subpixel():
    anarr = np.zeros((4, 5))
    anarr[2, 3] = 1
    # The correspondence principle should hold
    first_guess = (2, 3)
    second_guess = utils._interpolate(anarr, first_guess, rad=1)
    assert np.allclose(second_guess, (2, 3))

    # Now something more meaningful
    anarr[2, 4] = 1
    second_guess = utils._interpolate(anarr, first_guess, rad=1)
    assert np.allclose(second_guess, (2, 3.5))


def test_subpixel_edge():
    anarr = np.zeros((4, 5))
    anarr[3, 0] = 1
    anarr[3, 4] = 1
    first_guess = (4, 0)
    second_guess = utils._interpolate(anarr, first_guess, rad=2)
    assert np.allclose(second_guess, (3, -0.5))

    anarr[3, 0] += 1
    anarr[0, 4] = 1
    second_guess = utils._interpolate(anarr, first_guess, rad=2)
    assert np.allclose(second_guess, (3.25, -0.5))


def test_subpixel_crazy():
    anarr = np.zeros((4, 5))
    first_guess = (0, 0)
    second_guess = utils._interpolate(anarr, first_guess, rad=2)
    assert np.alltrue(second_guess < anarr.shape)
