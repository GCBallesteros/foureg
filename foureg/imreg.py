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
import numpy as np
import numpy.fft as fft
from scipy.ndimage import map_coordinates

import foureg.utils as utils

EXCESS_CONST = 1.1


def _logpolar_filter(shape):
    """
    Make a radial cosine filter for the logpolar transform.

    This filter suppresses low frequencies and completely removes the zero freq.
    """
    yy = np.linspace(-np.pi / 2.0, np.pi / 2.0, shape[0])[:, np.newaxis]
    xx = np.linspace(-np.pi / 2.0, np.pi / 2.0, shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy**2 + xx**2)
    filt = 1.0 - np.cos(rads) ** 2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1

    return filt


def _get_pcorr_shape(shape):
    ret = (int(max(shape) * 1.0),) * 2

    return ret


def _get_ang_scale(ims, exponent="inf", constraints=None):
    """
    Given two images, return their scale and angle difference.

    Parameters
    ----------
    ims (2-tuple-like of 2D ndarrays)
        The images
    exponent (float or 'inf')
        The exponent stuff, see :func:`similarity` constraints (dict, optional)

    Returns
    -------
    tuple: Scale, angle.

    Describes the relationship of the subject image to the first one.
    """
    assert len(ims) == 2, "Only two images are supported as input"
    shape = ims[0].shape

    ims_apod = [utils._apodize(im) for im in ims]
    dfts = [fft.fftshift(fft.fft2(im)) for im in ims_apod]
    filt = _logpolar_filter(shape)
    dfts = [dft * filt for dft in dfts]

    # High-pass filtering used to be here, but we have moved it to a higher
    # level interface

    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    stuffs = [_logpolar(np.abs(dft), pcorr_shape, log_base) for dft in dfts]

    (arg_ang, arg_rad), _ = _phase_correlation(
        stuffs[0],
        stuffs[1],
        utils.argmax_angscale,
        log_base,
        exponent,
        constraints,
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

    return scale, angle


def translation(im0, im1, filter_pcorr=0, odds=1, constraints=None):
    """
    Return translation vector to register images.

    It tells how to translate the im1 to get im0.

    Parameters
    ----------
    im0 (2D numpy array)
        The first (template) image
    im1 (2D numpy array)
        The second (subject) image
    filter_pcorr (int):
        Radius of the minimum spectrum filter for translation detection, use the
        filter when detection fails. Values > 3 are likely not useful.
    constraints (dict or None): Specify preference of seeked values.
        For more detailed documentation, refer to :func:`similarity`.
        The only difference is that here, only keys ``tx`` and/or ``ty``
        (i.e. both or any of them or none of them) are used.
    odds (float): The greater the odds are, the higher is the preferrence of the
        angle + 180 over the original angle. Odds of -1 are the same as inifinity.
        The value 1 is neutral, the converse of 2 is 1 / 2 etc.

    Returns
    -------
    dict: Contains following keys: ``angle``, ``tvec`` (Y, X), and ``success``.
    """
    # We estimate translation for the original image...
    tvec, succ = _translation(im0, im1, filter_pcorr, constraints)
    # ... and for the 180-degrees rotated image (the rotation estimation
    # doesn't distinguish rotation of x vs x + 180deg).
    tvec2, succ2 = _translation(im0, utils.rot180(im1), filter_pcorr, constraints)

    pick_rotated = succ2 * odds > succ or odds == -1
    if pick_rotated:
        ret = dict(tvec=tvec2, success=succ2, angle=180)
    else:
        ret = dict(tvec=tvec, success=succ, angle=0)

    return ret


def _get_precision(shape, scale=1):
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


def _similarity(
    im0,
    im1,
    numiter=1,
    order=3,
    constraints=None,
    filter_pcorr=0,
    exponent="inf",
    bgval=None,
):
    """
    This function takes some input and returns mutual rotation, scale and translation.

    It does these things during the process:
    * Handles correct constraints handling (defaults etc.).
    * Performs angle-scale determination iteratively.
      This involves keeping constraints in sync.
    * Performs translation determination.
    * Calculates precision.

    Returns
    -------
    Dictionary with results.
    """
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

    constraints_default = dict(angle=[0, None], scale=[1, None])
    if constraints is None:
        constraints = constraints_default

    # We guard against case when caller passes only one constraint key.
    # Now, the provided ones just replace defaults.
    constraints_default.update(constraints)
    constraints = constraints_default

    # During iterations, we have to work with constraints too.
    # So we make the copy in order to leave the original intact
    constraints_dynamic = constraints.copy()
    constraints_dynamic["scale"] = list(constraints["scale"])
    constraints_dynamic["angle"] = list(constraints["angle"])

    for _ in range(numiter):
        newscale, newangle = _get_ang_scale([im0, im2], exponent, constraints_dynamic)
        scale *= newscale
        angle += newangle

        constraints_dynamic["scale"][0] /= newscale
        constraints_dynamic["angle"][0] -= newangle

        transformation = similarity_matrix(scale, angle, (0, 0))
        im2 = transform_img(im1, transformation, bgval=bgval, order=order)

    # Here we look how is the turn-180
    target, stdev = constraints.get("angle", (0, None))
    odds = _get_odds(angle, target, stdev)

    # now we can use pcorr to guess the translation
    res = translation(im0, im2, filter_pcorr, odds, constraints)

    # The log-polar transform may have got the angle wrong by 180 degrees.
    # The phase correlation can help us to correct that
    angle += res["angle"]
    res["angle"] = utils.wrap_angle(angle, 360)

    Dangle, Dscale = _get_precision(shape, scale)

    affine = similarity_matrix(scale, res["angle"], res["tvec"])

    # 0.25 because we go subpixel now
    res = {"transformation": affine, "Dscale": Dscale, "Dangle": Dangle, "Dt": 0.25}

    return res


def similarity(
    im0,
    im1,
    numiter=1,
    order=3,
    constraints=None,
    filter_pcorr=0,
    exponent="inf",
):
    """
    Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic scale and
    without shear.

    Parameters
    ----------
    im0 (2D numpy array)
        The first (template) image
    im1 (2D numpy array)
        The second (subject) image
    numiter (int)
        How many times to iterate when determining scale and rotation
    order (int):
        Order of approximation (when doing transformations). 1 = linear, 3 = cubic etc.
    filter_pcorr (int):
        Radius of a spectrum filter for translation detection
    exponent (float or 'inf'):
        The exponent value used during processing. Refer to the docs for a thorough
        explanation. Generally, pass "inf" when feeling conservative. Otherwise,
        experiment, values below 5 are not even supposed to work.
    constraints (dict or None): Specify preference of seeked values.
        Pass None (default) for no constraints, otherwise pass a dict with
        keys ``angle``, ``scale``, ``tx`` and/or ``ty`` (i.e. you can pass
        all, some of them or none of them, all is fine). The value of a key
        is supposed to be a mutable 2-tuple (e.g. a list), where the first
        value is related to the constraint center and the second one to
        softness of the constraint (the higher is the number,
        the more soft a constraint is).

        More specifically, constraints may be regarded as weights in form of a shifted
        Gaussian curve. However, for precise meaning of keys and values, see the
        documentation section :ref:`constraints`. Names of dictionary keys map to
        names of command-line arguments.

    Returns
    -------
    dict: Contains following keys: ``scale``, ``angle``, ``tvec`` (Y, X),
    ``success`` and ``timg`` (the transformed subject image)

    Note
    ----
    There are limitations

    * Scale change must be less than 2.
    * No subpixel precision (but you can use *resampling* to get around this).
    """
    bgval = utils.get_borderval(im1, 5)

    res = _similarity(
        im0, im1, numiter, order, constraints, filter_pcorr, exponent, bgval
    )
    print(res)

    im2 = transform_img(im1, res["transformation"], bgval, order)
    # Order of mask should be always 1 - higher values produce strange results.
    imask = transform_img(np.ones_like(im1), res["transformation"], 0, 1)
    # This removes some weird artifacts
    imask[imask > 0.8] = 1.0

    # Framing here = just blending the im2 with its BG according to the mask
    im3 = utils.frame_img(im2, imask, 10)

    res["timg"] = im3
    return res


def _get_odds(angle, target, stdev):
    """
    Determine whether we are more likely to choose the angle, or angle + 180°

    Parameters
    ----------
    angle (float, degrees)
        The base angle.
    target (float, degrees)
        The angle we think is the right one. Typically, we take this from constraints.
    stdev (float, degrees)
        The relevance of the target value, typically taken from constraints.

    Returns
    -------
    float: The greater the odds are, the higher is the preferrence
        of the angle + 180 over the original angle. Odds of -1 are the same
        as inifinity.
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


def _translation(im0, im1, filter_pcorr=0, constraints=None):
    """
    The plain wrapper for translation phase correlation, no big deal.
    """
    # Apodization and pcorr don't play along
    # im0, im1 = [utils._apodize(im, ratio=1) for im in (im0, im1)]
    ret, succ = _phase_correlation(
        im0, im1, utils.argmax_translation, filter_pcorr, constraints
    )
    return ret, succ


def _phase_correlation(im0, im1, callback=None, *args):
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
    if callback is None:
        callback = utils._argmax2D

    # TODO: Implement some form of high-pass filtering of PHASE correlation
    f0, f1 = [fft.fft2(arr) for arr in (im0, im1)]
    # spectrum can be filtered (already),
    # so we have to take precaution against dividing by 0
    eps = abs(f1).max() * 1e-15
    # cps == cross-power spectrum of im0 and im1
    cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    # scps = shifted cps
    scps = fft.fftshift(cps)

    (t0, t1), success = callback(scps, *args)
    ret = np.array((t0, t1))

    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= np.array(f0.shape, int) // 2
    return ret, success


def transform_img(
    img, transformation, bgval=None, order=1, mode="constant", invert=False
):
    """
    Return translation vector to register images.

    Parameters
    ----------
    img (2D or 3D numpy array)
        What will be transformed.
        If a 3D array is passed, it is treated in a manner in which RGB
        images are supposed to be handled - i.e. assume that coordinates
        are (Y, X, channels).
        Complex images are handled in a way that treats separately
        the real and imaginary parts.
    scale (float)
        The scale factor (scale > 1.0 means zooming in)
    angle (float)
        Degrees of rotation (clock-wise)
    tvec (2-tuple)
        Pixel translation vector, Y and X component.
    mode (string)
        The transformation mode (refer to e.g.
        :func:`scipy.ndimage.shift` and its kwarg ``mode``).
    bgval (float)
        Shade of the background (filling during transformations)
        If None is passed, :func:`imreg_dft.utils.get_borderval` with
        radius of 5 is used to get it.
    order (int)
        Order of approximation (when doing transformations). 1 =
        linear, 3 = cubic etc. Linear works surprisingly well.

    Returns
    -------
    np.ndarray: The transformed img, may have another
    i.e. (bigger) shape than the source.
    """
    if invert:
        transformation = np.linalg.inv(transformation)

    if bgval is None:
        bgval = utils.get_borderval(img)

    if img.ndim == 3:
        # A bloody painful special case of RGB images
        ret = np.empty_like(img)
        for idx in range(img.shape[2]):
            sli = (slice(None), slice(None), idx)
            ret[sli] = transform_img(img[sli], transformation, bgval, order, mode)
        return ret

    transformed_coords = utils.transform_2d_coord_arrays(
        np.linalg.inv(transformation),
        np.dstack(
            np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
        ),
    )
    n_pixels = np.prod(img.shape)
    transformed_coords = transformed_coords.reshape(n_pixels, 2)
    slave_transformed = map_coordinates(img, transformed_coords.T, cval=bgval)
    slave_transformed = np.reshape(slave_transformed, img.shape)

    dest = slave_transformed

    return dest


def similarity_matrix(scale, angle, tvec):
    # Rotation
    angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    r_matrix = np.identity(3)
    r_matrix[:2, :2] = np.array([[c, -s], [s, c]])

    # Scaling
    s_matrix = np.identity(3)
    s_matrix[0, 0] = scale
    s_matrix[1, 1] = scale

    # Translation
    t_matrix = np.identity(3)
    t_matrix[:2, 2] = tvec

    return t_matrix @ r_matrix @ s_matrix


def _get_log_base(shape, new_r):
    """
    Basically common functionality of `_logpolar` and `_get_ang_scale`

    This value can be considered fixed, if you want to mess with the logpolar
    transform, mess with the shape.

    Parameters
    ----------
    shape
        Shape of the original image.
    new_r: float
        The r-size of the log-polar transform array dimension.

    Returns
    -------
    float: Base of the log-polar transform.
    The following holds:
    `log\_base =
        \exp( \ln [ \mathit{spectrum\_dim} ] / \mathit{loglpolar\_scale\_dim} )`,
    or the equivalent
        `log\_base^{\mathit{loglpolar\_scale\_dim}} = \mathit{spectrum\_dim}`.
    """
    # The highest radius we have to accomodate is 'old_r',
    # However, we cut some parts out as only a thin part of the spectra has
    # these high frequencies
    old_r = shape[0] * EXCESS_CONST
    # We are radius, so we divide the diameter by two.
    old_r /= 2.0
    # we have at most 'new_r' of space.
    log_base = np.exp(np.log(old_r) / new_r)
    return log_base


def _logpolar(image, shape, log_base, bgval=None):
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
        bgval = np.percentile(image, 1)
    imshape = np.array(image.shape)
    center = imshape[0] / 2.0, imshape[1] / 2.0
    # 0 .. pi = only half of the spectrum is used
    theta = utils._get_angles(shape)
    radius_x = utils._get_lograd(shape, log_base)
    radius_y = radius_x.copy()
    ellipse_coef = imshape[0] / float(imshape[1])
    # We have to acknowledge that the frequency spectrum can be deformed
    # if the image aspect ratio is not 1.0
    # The image is x-thin, so we acknowledge that the frequency spectra
    # scale in x is shrunk.
    radius_x /= ellipse_coef

    y = radius_y * np.sin(theta) + center[0]
    x = radius_x * np.cos(theta) + center[1]
    output = np.empty_like(y)
    map_coordinates(image, [y, x], output=output, order=3, mode="constant", cval=bgval)

    return output
