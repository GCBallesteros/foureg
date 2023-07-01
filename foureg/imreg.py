# imreg.py

# Copyright (c) 2014-?, Matěj Týč
# Copyright (c) 2011-2014, Christoph Gohlke
# Copyright (c) 2011-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
FFT based image registration. --- main functions
"""
from copy import deepcopy
from typing import Any, Callable, Optional

import numpy as np
import torch

import foureg.utils as utils

from .constraints import Constraints
from .result import Result

EXCESS_CONST = 1.1
PI = torch.pi


def _logpolar_filter(shape: tuple[int, int]):
    """
    Make a radial cosine filter for the logpolar transform.

    This filter suppresses low frequencies and completely removes the zero freq.
    """
    yy = torch.linspace(-PI / 2.0, PI / 2.0, shape[0])[:, None]
    xx = torch.linspace(-PI / 2.0, PI / 2.0, shape[1])[None, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = torch.sqrt(yy**2 + xx**2)
    filt = 1.0 - torch.cos(rads) ** 2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[torch.abs(rads) > PI / 2] = 1

    return filt


def _get_pcorr_shape(shape: tuple[int, int]) -> tuple[int, int]:
    ret = (int(max(shape) * 1.0),) * 2

    return ret


def _get_ang_scale(
    ims: list[torch.Tensor],
    exponent: float = torch.inf,
    constraints: Constraints = Constraints(),
) -> tuple[float, float]:
    """
    Given two images, return their scale and angle difference.

    Parameters
    ----------
    ims
        The images
    exponent
        The exponent stuff, see :func:`similarity` constraints (dict, optional)

    Returns
    -------
    Scale and angle tuple
    """
    assert len(ims) == 2, "Only two images are supported as input"
    shape = ims[0].shape

    ims_apod = [utils.apodize(im) for im in ims]
    dfts = [torch.fft.fftshift(torch.fft.fft2(im)) for im in ims_apod]
    filt = _logpolar_filter(shape)
    dfts = [dft * filt for dft in dfts]

    # High-pass filtering used to be here, but we have moved it to a higher
    # level interface

    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    stuffs = [_logpolar(torch.abs(dft), pcorr_shape, log_base) for dft in dfts]

    (arg_ang, arg_rad), _ = _phase_correlation(
        stuffs[0],
        stuffs[1],
        lambda img: utils.argmax_angscale(img, log_base, exponent, constraints),
    )

    angle = -np.pi * arg_ang / float(pcorr_shape[0])
    angle = np.rad2deg(angle)
    angle = utils.wrap_angle(angle, 360)
    scale = log_base**arg_rad

    angle = -angle
    scale = 1.0 / scale

    if not 0.5 < scale < 2:
        raise ValueError(
            "Images are not compatible. Scale change %g too big to be true." % scale
        )

    return scale.item(), angle.item()


def translation(
    im0: torch.Tensor,
    im1: torch.Tensor,
    constraints: Constraints,
    filter_pcorr: int = 0,
    odds: float = 1,
) -> dict[str, Any]:
    """
    Return translation vector to register images.

    It tells how to translate the im1 to get im0.

    Parameters
    ----------
    im0
        The first (template) image
    im1
        The second (subject) image
    filter_pcorr
        Radius of the minimum spectrum filter for translation detection, use the
        filter when detection fails. Values > 3 are likely not useful.
    constraints
        Specify preference of seeked values.
        For more detailed documentation, refer to :func:`similarity`.
        The only difference is that here, only keys ``tx`` and/or ``ty``
        (i.e. both or any of them or none of them) are used.
    odds
        The greater the odds are, the higher is the preferrence of the
        angle + 180 over the original angle. Odds of -1 are the same as inifinity.
        The value 1 is neutral, the converse of 2 is 1 / 2 etc.

    Returns
    -------
    A dict that contains following keys: ``angle``, ``tvec`` (Y, X), and ``success``.
    """
    # We estimate translation for the original image...
    tvec, succ = _translation(im0, im1, constraints, filter_pcorr)
    # ... and for the 180-degrees rotated image (the rotation estimation
    # doesn't distinguish rotation of x vs x + 180deg).
    tvec2, succ2 = _translation(im0, utils.rot180(im1), constraints, filter_pcorr)

    pick_rotated = succ2 * odds > succ or odds == -1
    if pick_rotated:
        ret = dict(tvec=tvec2, success=succ2, angle=180)
    else:
        ret = dict(tvec=tvec, success=succ, angle=0)

    return ret


def _get_precision(shape: tuple[int, int], scale: float = 1) -> tuple[float, float]:
    """
    Given the parameters of the log-polar transform, get width of the interval
    where the correct values are.

    Parameters
    ----------
    shape (tuple)
        Shape of images
    scale (float)
        The scale difference (precision varies)
    """
    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    # * 0.5 <= max deviation is half of the step
    # * 0.25 <= we got subpixel precision now and 0.5 / 2 == 0.25
    # sccale: Scale deviation depends on the scale value
    Dscale = scale * (log_base - 1) * 0.25
    # angle: Angle deviation is constant
    Dangle = 180.0 / pcorr_shape[0] * 0.25
    return Dangle, Dscale


def similarity(
    im0: torch.Tensor,
    im1: torch.Tensor,
    numiter: int = 1,
    mode: str = "bilinear",
    constraints: Constraints = Constraints(),
    filter_pcorr: int = 0,
    exponent: float = torch.inf,
) -> Result:
    """
    Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    It does these things during the process:
    * Handles correct constraints handling (defaults etc.).
    * Performs angle-scale determination iteratively.
      This involves keeping constraints in sync.
    * Performs translation determination.
    * Calculates precision.

    A similarity transformation is an affine transformation with isotropic scale and
    without shear.

    Parameters
    ----------
    im0
        The first (template) image
    im1
        The second (subject) image
    numiter
        How many times to iterate when determining scale and rotation
    mode
        Interpolation mode passed to grid_sample
    filter_pcorr
        Radius of a spectrum filter for translation detection
    exponent
        The exponent value used during processing. Refer to the docs for a thorough
        explanation. Generally, pass "inf" when feeling conservative. Otherwise,
        experiment, values below 5 are not even supposed to work.
    constraint
        Specify preference of seeked values.
        Pass None (default) for no constraints, otherwise pass a dict with
        keys ``angle``, ``scale``, ``tx`` and/or ``ty`` (i.e. you can pass
        all, some of them or none of them, all is fine). The value of a key
        is supposed to be a mutable 2-tuple (e.g. a list), where the first
        value is related to the constraint center and the second one to
        softness of the constraint (the higher is the number,
        the more soft a constraint is).

        More specifically, constraints may be regarded as weights in form of a shifted
        Gaussian curve. However, for precise meaning of keys and values, see the
        documentation section `constraints`. Names of dictionary keys map to
        names of command-line arguments.

    Returns
    -------
    Similarity transformation and associated uncertainties

    Note
    ----
    There are limitations

    * Scale change must be less than 2.
    * No subpixel precision (but you can use *resampling* to get around this).
    """
    im1 = torch.Tensor(im1).type(torch.float32)
    im0 = torch.Tensor(im0).type(torch.float32)

    bgval = utils.get_borderval(im1, 5)
    # ims_torch = [torch.Tensor(img.copy()).type(torch.float32) for img in ims]
    if bgval is None:
        bgval = utils.get_borderval(im1, 5)

    shape = im0.shape
    if shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif im0.ndim != 2:
        raise ValueError("Images must be 2-dimensional.")

    # We are going to iterate and precise scale and angle estimates
    scale = 1.0
    angle = 0.0
    im2 = im1

    # get a copy of the constraints so that we can modify them during the iterations
    constraints_dynamic = deepcopy(constraints)

    for _ in range(numiter):
        newscale, newangle = _get_ang_scale([im0, im2], exponent, constraints_dynamic)
        scale *= newscale
        angle += newangle

        constraints_dynamic.scale = (
            constraints_dynamic.scale[0] / newscale,
            constraints_dynamic.scale[1],
        )
        constraints_dynamic.angle = (
            constraints_dynamic.angle[0] + newangle,
            constraints_dynamic.angle[1],
        )

        # transform_img understand the angle the other way around hence the minus
        # sign below
        transformation = utils.similarity_matrix(scale, -angle, (0, 0))

        # False because we're transforming the slave to be closer to the master
        im2 = utils.transform_img(
            im1, transformation, bgval=bgval, mode=mode, invert=False
        )

    # Here we look how is the turn-180
    target, stdev = constraints.angle
    odds = _get_odds(angle, target, stdev)

    # now we can use pcorr to guess the translation
    res = translation(im0, im2, constraints, filter_pcorr, odds)

    # Flipping tvec is required because transform_img has a different
    # understanding of what is xy wrt the output of translation
    res["tvec"] = res["tvec"].flip(dims=(0,))

    # The log-polar transform may have got the angle wrong by 180 degrees.
    # The phase correlation can help us to correct that
    angle += res["angle"]
    res["angle"] = utils.wrap_angle(angle, 360)

    Dangle, Dscale = _get_precision(shape, scale)

    # transform_img understand the angle the other way around hence the minus sign
    # below
    # dt 0.25 because we go subpixel now (???)
    res = Result(
        angle=-res["angle"],
        scale=scale,
        tvec=res["tvec"],
        dscale=Dscale,
        dangle=Dangle,
        dt=0.25,
    )

    return res


def _get_odds(angle: float, target: float, stdev: float) -> float:
    """
    Determine whether we are more likely to choose the angle, or angle + 180°

    Parameters
    ----------
    angle
        The base angle in degrees.
    target
        The angle we think is the right one. Typically, we take this from constraints.
        In degrees.
    stdev
        The relevance of the target value, typically taken from constraints.

    Returns
    -------
    float: The greater the odds are, the higher is the preferrence of the angle + 180
        ver the original angle. Odds of -1 are the same as inifinity.
    """
    ret = 1
    if stdev is not None:
        diffs = [
            abs(utils.wrap_angle(ang, 360))
            for ang in (target - angle, target - angle + 180)
        ]
        odds0, odds1 = 0, 0
        if stdev > 0:
            odds0, odds1 = [np.exp(-(diff**2) / stdev**2) for diff in diffs]
        if odds0 == 0 and odds1 > 0:
            # -1 is treated as infinity in _translation
            ret = -1
        elif stdev == 0 or (odds0 == 0 and odds1 == 0):
            ret = -1
            if diffs[0] < diffs[1]:
                ret = 0
        else:
            ret = odds1 / odds0

    return ret


def _translation(
    im0: torch.Tensor,
    im1: torch.Tensor,
    constraints: Constraints,
    filter_pcorr: int = 0,
) -> tuple[torch.Tensor, Any]:
    """
    The plain wrapper for translation phase correlation, no big deal.
    """
    # Apodization and pcorr don't play along
    ret, succ = _phase_correlation(
        im0,
        im1,
        lambda img: utils.argmax_translation(img, filter_pcorr, constraints),
    )
    return ret, succ


def _phase_correlation(
    im0: torch.Tensor,
    im1: torch.Tensor,
    callback: Callable[[torch.Tensor], Any],
) -> tuple[torch.Tensor, Any]:
    """
    Computes phase correlation between im0 and im1

    Args:
    Parameters
    ----------
    im0
    im1
    callback (function)
        Process the cross-power spectrum (i.e. choose coordinates of the best element,
        usually of the highest one). Defaults to :func:`imreg_dft.utils.argmax2D`

    Returns
    -------
    tuple: The translation vector (Y, X). Translation vector of (0, 0)
    means that the two images match.
    """
    # TODO: Implement some form of high-pass filtering of PHASE correlation
    f0, f1 = [torch.fft.fft2(arr) for arr in (im0, im1)]
    # spectrum can be filtered (already),
    # so we have to take precaution against dividing by 0
    eps = torch.abs(f1).max() * 1e-15
    # cps == cross-power spectrum of im0 and im1
    cps = torch.abs(torch.fft.ifft2((f0 * f1.conj()) / (abs(f0) * abs(f1) + eps)))
    # scps = shifted cps
    scps = torch.fft.fftshift(cps)

    (t0, t1), success = callback(scps)
    ret = torch.tensor((t0, t1))

    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= torch.tensor(f0.shape) // 2
    return ret, success


def _get_log_base(shape: tuple[int, int], new_r: float) -> float:
    """
    Basically common functionality of `_logpolar` and `_get_ang_scale`

    This value can be considered fixed, if you want to mess with the logpolar
    transform, mess with the shape.

    Parameters
    ----------
    shape
        Shape of the original image.
    new_r
        The r-size of the log-polar transform array dimension.

    Returns
    -------
    Base of the log-polar transform.
    The following holds:
    `log\\_base =
        \\exp( \\ln [ \\mathit{spectrum\\_dim} ] / \\mathit{loglpolar\\_scale\\_dim} )`,
    or the equivalent
        `log\\_base^{\\mathit{loglpolar\\_scale\\_dim}} = \\mathit{spectrum\\_dim}`.
    """
    import math

    # The highest radius we have to accomodate is 'old_r',
    # However, we cut some parts out as only a thin part of the spectra has
    # these high frequencies
    old_r = shape[0] * EXCESS_CONST
    # We are radius, so we divide the diameter by two.
    old_r /= 2.0
    # we have at most 'new_r' of space.
    log_base = math.exp(math.log(old_r) / new_r)
    return log_base


def _logpolar(
    image: torch.Tensor,
    shape: tuple[int, int],
    log_base: float,
    bgval: Optional[float] = None,
) -> torch.Tensor:
    """
    Return log-polar transformed image.
    Takes into account anisotropicity of the freq spectrum of rectangular images

    Parameters
    ----------
    image
        The image to be transformed
    shape
        Shape of the transformed image
    log_base
        Parameter of the transformation, get it via `_get_log_base`
    bgval
        The backround value. If None, use minimum of the image.

    Returns
    -------
    The transformed image
    """
    if bgval is None:
        bgval = torch.quantile(image, 0.99).item()
    imshape = image.shape
    center = imshape[0] / 2.0, imshape[1] / 2.0
    # 0 .. pi = only half of the spectrum is used
    theta = utils._get_angles(shape)
    radius_x = utils._get_lograd(shape, log_base)
    radius_y = radius_x.clone()
    ellipse_coef = imshape[0] / float(imshape[1])
    # We have to acknowledge that the frequency spectrum can be deformed
    # if the image aspect ratio is not 1.0
    # The image is x-thin, so we acknowledge that the frequency spectra
    # scale in x is shrunk.
    radius_x /= ellipse_coef

    y = radius_y * torch.sin(theta) + center[0]
    x = radius_x * torch.cos(theta) + center[1]

    yx = torch.Tensor(torch.dstack([x, y]))
    output_t = utils.map_coordinates(yx, image, bgval)

    return output_t
