import numpy as np

import foureg.imreg as imreg


def test_odds():
    # low or almost-zero odds
    odds = imreg._get_odds(10, 20, 0)
    assert np.isclose(odds, 0)

    odds = imreg._get_odds(10, 20, 0.1)
    assert np.isclose(odds, 0)

    odds = imreg._get_odds(10, 20, 40)
    assert odds < 0.01

    # non-zero complementary odds
    odds = imreg._get_odds(10, 20, 100)
    assert odds < 0.6

    odds = imreg._get_odds(10, 200, 100)
    assert odds > 1 / 0.6

    # high (near-infinity) odds
    odds = imreg._get_odds(10, 200, 0)
    assert odds == -1

    odds = imreg._get_odds(10, 200, 0.1)
    assert odds == -1
