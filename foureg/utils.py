# utils.py

# Copyright (c) 2014-?, Matěj Týč
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
FFT based image registration. --- utility functions
"""
import math
from typing import Iterable, Optional

import numpy as np
import numpy.fft as fft
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .constraints import Constraints


def wrap_angle(angles, ceil=2 * np.pi):
    """
    Parameters
    ----------
    angles (float or ndarray, unit depends on kwarg ``ceil``)
    ceil (float)
        Turnaround value
    """
    angles += ceil / 2.0
    angles %= ceil
    angles -= ceil / 2.0
    return angles


def rot180(arr: torch.Tensor) -> torch.Tensor:
    """
    Rotate the input array over 180°
    """
    ret = torch.rot90(arr, 2)
    return ret


def _get_angles(shape: tuple[int, int]) -> torch.Tensor:
    """
    In the log-polar spectrum, the (first) coord corresponds to an angle.
    This function returns a mapping of (the two) coordinates
    to the respective angle.
    """
    ret = torch.zeros(shape, dtype=torch.float32)
    ret -= torch.linspace(0, torch.pi, shape[0])[:, None]
    return ret


def _get_lograd(shape: tuple[int, int], log_base: float) -> torch.Tensor:
    """
    In the log-polar spectrum, the (second) coord corresponds to an angle.
    This function returns a mapping of (the two) coordinates
    to the respective scale.

    Returns
    -------
    2D np.ndarray of shape ``shape``, -1 coord contains scales
    from 0 to log_base ** (shape[1] - 1)
    """
    ret = torch.zeros(shape, dtype=torch.float32)
    ret += torch.pow(log_base, torch.arange(shape[1], dtype=torch.float32))[None, :]
    return ret


def _get_constraint_mask(shape, log_base, constraints: Constraints):
    """
    Prepare mask to apply to constraints to a cross-power spectrum.
    """
    mask = torch.ones(shape, dtype=torch.float32)

    # Here, we create masks that modulate picking the best correspondence.
    # Generally, we look at the log-polar array and identify mapping of
    # coordinates to values of quantities.

    ## Apply SCALE constraints
    scale, sigma = constraints.scale
    if not math.isnan(sigma):
        scales = torch.fft.ifftshift(_get_lograd(shape, log_base))
        # vvv This issome kind of transformation of result of _get_lograd
        # vvv (log radius in pixels) to the linear scale.
        scales *= log_base ** (-shape[1] / 2.0)
        # This makes the scales array low near where scales is near 'scale'
        scales -= 1.0 / scale

        if sigma == 0:
            # there isn't: ascales = np.abs(scales - scale)
            # because scales are already low for values near 'scale'
            ascales = torch.abs(scales)
            scale_min = ascales.min()
            mask[ascales > scale_min] = 0
        else:
            mask *= torch.exp(-(scales**2) / sigma**2)

    ## Apply ANGLE constraints
    angle, sigma = constraints.angle
    if not math.isnan(sigma):
        angles = _get_angles(shape)
        # We flip the sign on purpose
        # TODO: ^^^ Why???
        angles -= np.deg2rad(angle)
        # TODO: Check out the wrapping. It may be tricky since pi+1 != 1
        angles = wrap_angle(angles, torch.pi)
        angles = torch.rad2deg(angles)
        if sigma == 0:
            aangles = torch.abs(angles)
            angle_min = aangles.min()
            mask[aangles > angle_min] = 0
        else:
            mask *= torch.exp(-(angles**2) / sigma**2)

    return torch.fft.fftshift(mask)


def argmax_angscale(
    array: torch.Tensor, log_base, exponent: float, constraints: Constraints
) -> tuple[torch.Tensor, float]:
    """
    Given a power spectrum, we choose the best fit.

    The power spectrum is treated with constraint masks and then
    passed to `_argmax_ext`.
    """
    mask = _get_constraint_mask(array.shape, log_base, constraints)
    array_orig = array.clone()

    array *= mask
    ret = _argmax_ext(array, exponent)
    ret_final = _interpolate(array, ret)

    success = _get_success(array_orig, tuple(ret_final), 0)

    return ret_final, success.item()


def min_filter_torch(arr: torch.Tensor, kernel_size: int) -> torch.Tensor:
    arr = arr.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    output = -F.max_pool2d(
        F.pad(-arr, (pad, pad, pad, pad), value=-torch.inf), kernel_size, stride=1
    ).squeeze()

    return output


def argmax_translation(
    array: torch.Tensor, filter_pcorr: int, constraints: Constraints
):
    # We want to keep the original and here is obvious that
    # it won't get changed inadvertently
    array_orig = array.clone()
    if filter_pcorr > 0:
        array = min_filter_torch(array, filter_pcorr)

    ashape = array.shape
    mask = torch.ones(array.shape, dtype=torch.float32)
    # first goes Y, then X
    for dim, key in enumerate(["ty", "tx"]):
        if math.isnan(constraints.__getattribute__(key)[1]):
            continue
        pos, sigma = constraints.__getattribute__(key)
        alen = ashape[dim]
        dom = torch.as_tensor(np.linspace(-alen // 2, -alen // 2 + alen, alen, False))
        if sigma == 0:
            # generate a binary array closest to the position
            idx = torch.argmin(torch.abs(dom - pos))
            vals = torch.zeros_like(dom)
            vals[idx] = 1.0
        else:
            vals = torch.exp(-((dom - pos) ** 2) / sigma**2)

        if dim == 0:
            mask *= vals[:, np.newaxis]
        else:
            mask *= vals[np.newaxis, :]

    array *= mask

    # WE ARE FFTSHIFTED already.
    # ban translations that are too big
    aporad = min(ashape[0] // 6, ashape[1] // 6)
    mask2 = get_apofield(ashape, aporad)
    array *= mask2
    # Find what we look for
    tvec = _argmax_ext(array, torch.inf)
    tvec = _interpolate(array_orig, tvec)

    # If we use constraints or min filter,
    # array_orig[tvec] may not be the maximum
    success = _get_success(array_orig, tuple(tvec), 2)

    return tvec, success


def _get_success(array, coord, radius=2):
    """
    Given a coord, examine the array around it and return a number signifying how good
    is the "match".

    Parameters
    ----------
    radius
        Get the success as a sum of neighbor of coord of this radius
    coord
        Coordinates of the maximum. Float numbers are allowed (and converted to int
        inside)

    Returns
    -------
    Success as float between 0 and 1 (can get slightly higher than 1).
    The meaning of the number is loose, but the higher the better.
    """
    coord = np.round(coord).astype(int)
    coord = tuple(coord)

    subarr = _get_subarr(array, coord, radius)

    theval = subarr.sum()
    theval2 = array[coord]
    # bigval = np.percentile(array, 97)
    # success = theval / bigval
    # TODO: Think this out
    success = torch.sqrt(theval * theval2)
    return success


def _argmax2D(array):
    """
    Simple 2D argmax function with simple sharpness indication
    """
    amax = np.argmax(array)
    ret = list(np.unravel_index(amax, array.shape))

    return np.array(ret)


def _get_subarr(array, center, rad):
    """
    Get a subarray around a cell in the array with wrapping around the
    borders of the image.

    Parameters
    ----------
    array (ndarray)
        The array to search
    center (2-tuple)
        The point in the array to search around
    rad (int)
        Search radius, no radius (i.e. get the single point) implies rad == 0
    """
    tarray = torch.nn.functional.pad(
        array.unsqueeze(0).unsqueeze(0),
        pad=(rad, rad, rad, rad),
        mode="circular",
    ).squeeze()

    # The center has to move due to the padding
    center_ = (center[0] + rad, center[1] + rad)

    subarray = tarray[
        center_[0] - rad : center_[0] + rad + 1,
        center_[1] - rad : center_[1] + rad + 1,
    ]
    return subarray


def _interpolate(array: torch.Tensor, rough, rad: int = 2):
    """
    Returns index that is in the array after being rounded.

    The result index tuple is in each of its components between zero and the
    array's shape.
    """
    rough = torch.round(rough).type(torch.int64)
    surroundings = _get_subarr(array, rough, rad)
    com = _argmax_ext(surroundings, 1)
    offset = com - rad
    ret = rough + offset
    # similar to win.wrap, so
    # -0.2 becomes 0.3 and then again -0.2, which is rounded to 0
    # -0.8 becomes - 0.3 -> len() - 0.3 and then len() - 0.8,
    # which is rounded to len() - 1. Yeah!
    ret += 0.5
    ret %= np.array(array.shape).astype(int)
    ret -= 0.5
    return ret


def _argmax_ext(array: torch.Tensor, exponent: float) -> torch.Tensor:
    """
    Calculate coordinates of the COM (center of mass) of the provided array.

    Parameters
    ----------
    array
        The array to be examined.
    exponent
        The exponent we power the array with. If the value 'inf' is given,
         the coordinage of the array maximum is taken.

    Returns
    -------
    The COM coordinate tuple, float values are allowed!
    """

    # When using an integer exponent for _argmax_ext, it is good to have the
    # neutral rotation/scale in the center rather near the edges

    if exponent == torch.inf:
        ret = _argmax2D(array)
    else:
        col = torch.arange(array.shape[0])[:, np.newaxis]
        row = torch.arange(array.shape[1])[np.newaxis, :]

        arr2 = array**exponent
        arrsum = arr2.sum()
        if arrsum == 0:
            # We have to return SOMETHING, so let's go for (0, 0)
            return torch.zeros(2)
        arrprody = torch.sum(arr2 * col) / arrsum
        arrprodx = torch.sum(arr2 * row) / arrsum
        ret = [arrprody, arrprodx]
        # We don't use it, but it still tells us about value distribution

    return torch.tensor(ret)


def imfilter(img, low=None, high=None, cap=None):
    """
    Given an image, it a high-pass and/or low-pass filters on its
    Fourier spectrum.

    Parameters
    ----------
    img (ndarray)
        The image to be filtered
    low (tuple)
        The low-pass filter parameters, 0..1
    high (tuple)
        The high-pass filter parameters, 0..1
    cap (tuple)
        The quantile cap parameters, 0..1. A filtered image will have extremes below
        the lower quantile and above the upper one cut.

    Returns
    -------
    np.ndarray
    The real component of the image after filtering
    """
    dft = fft.fft2(img)

    if low is not None:
        _lowpass(dft, low[0], low[1])
    if high is not None:
        _highpass(dft, high[0], high[1])

    ret = fft.ifft2(dft)
    # if the input was a real number array, return real numbers,
    # otherwise let it be complex.
    if not np.iscomplexobj(img):
        ret = np.real(ret)

    if cap is None:
        cap = (0, 1)

    low, high = cap
    if low > 0.0:
        low_val = np.percentile(ret, low * 100.0)
        ret[ret < low_val] = low_val
    if high < 1.0:
        high_val = np.percentile(ret, high * 100.0)
        ret[ret > high_val] = high_val

    return ret


def _highpass(dft, lo, hi):
    mask = _xpass((dft.shape), lo, hi)
    dft *= 1 - mask


def _lowpass(dft, lo, hi):
    mask = _xpass((dft.shape), lo, hi)
    dft *= mask


def _xpass(shape, lo, hi):
    """
    Compute a pass-filter mask with values ranging from 0 to 1.0
    The mask is low-pass, application has to be handled by a calling funcion.
    """
    assert lo <= hi, "Filter order wrong, low '%g', high '%g'" % (lo, hi)
    assert lo >= 0, "Low filter lower than zero (%g)" % lo
    # High can be as high as possible

    dom_x = np.fft.fftfreq(shape[0])[:, np.newaxis]
    dom_y = np.fft.fftfreq(shape[1])[np.newaxis, :]

    # freq goes 0..0.5, we want from 0..1, so we multiply it by 2.
    dom = np.sqrt(dom_x**2 + dom_y**2) * 2

    res = np.ones(dom.shape)
    res[dom >= hi] = 0.0
    mask = (dom > lo) * (dom < hi)
    res[mask] = 1 - (dom[mask] - lo) / (hi - lo)

    return res


def apodize(img: torch.Tensor) -> torch.Tensor:
    """
    Given an image, it apodizes it (so it becomes quasi-seamless).
    Color near the edges will converge to the same color

    Parameters
    ----------
    img
        Input img

    Returns
    -------
        The apodized image
    """

    mindim = min(img.shape)
    aporad = int(mindim * 0.12)
    apofield = get_apofield(img.shape, aporad)
    res = img * apofield
    bg = get_borderval(img, aporad // 2)
    res += bg * (1 - apofield)

    return res


def get_apofield(shape, aporad: int):
    """
    Returns an array between 0 and 1 that goes to zero close to the edges.
    """
    if aporad == 0:
        return np.ones(shape, dtype=float)
    apos = np.hanning(aporad * 2)
    vecs = []
    for dim in shape:
        assert dim > aporad * 2, "Apodization radius %d too big for shape dim. %d" % (
            aporad,
            dim,
        )
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad:] = apos[-aporad:]
        vecs.append(toapp)
    apofield = np.outer(vecs[0], vecs[1])

    return apofield.astype(np.float32)


def gaussian_kernel_1d(sigma: float, num_sigma: float = 3.0) -> torch.Tensor:
    radius = math.ceil(sigma * num_sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()

    return kernel.mul_(1 / kernel.sum())


def gaussian_filter(
    img: torch.Tensor, sigma: float, mode: str = "circular"
) -> torch.Tensor:
    kernel_1d = gaussian_kernel_1d(sigma)
    padding = len(kernel_1d) // 2
    img = F.pad(
        img.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode=mode
    )
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1))
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1))

    return img.squeeze(0).squeeze(0)


def frame_img(img, transform, dst, apofield=None):
    """
    Given an array, a mask (floats between 0 and 1), and a distance, alter the area
    where the mask is low (and roughly within dst from the edge) so it blends well
    with the area where the mask is high. The purpose of this is removal of spurious
    frequencies in the image's Fourier spectrum.

    Parameters
    ----------
    img
        What we want to alter
    maski
        The indicator what can be altered (0) and what can not (1)
    dst
        Parameter controlling behavior near edges, value could be probably deduced
        from the mask.
    """
    # Order of mask should be always 1 - higher values produce strange results.
    mask = transform_img(torch.ones_like(img), transform, 0, "nearest", invert=False)

    # This removes some weird artifacts
    mask[mask > 0.8] = 1.0

    radius = dst / 1.8

    convmask0 = mask + 1e-10

    krad_max = radius * 6
    convimg = img
    convmask = convmask0
    convimg0 = img
    krad0 = 0.8
    krad = krad0

    while krad < krad_max:
        convimg = gaussian_filter(convimg0 * convmask0, krad, mode="circular")
        convmask = gaussian_filter(convmask0, krad, mode="circular")
        convimg /= convmask

        convimg = convimg * (convmask - convmask0) + convimg0 * (
            1 - convmask + convmask0
        )
        krad *= 1.8

        convimg0 = convimg
        convmask0 = convmask

    if apofield is not None:
        ret = convimg * (1 - apofield) + img * apofield
    else:
        ret = convimg
        ret[mask >= 1] = img[mask >= 1]

    return ret


def get_borderval(img: torch.Tensor, radius: Optional[int] = None) -> float:
    """
    Given an image and a radius, examine the average value of the image
    at most radius pixels from the edge
    """
    if radius is None:
        mindim = min(img.shape)
        radius = max(1, mindim // 20)
    mask = torch.zeros_like(img, dtype=torch.bool)
    mask[:, :radius] = True
    mask[:, -radius:] = True
    mask[:radius, :] = True
    mask[-radius:, :] = True

    median = torch.median(img[mask]).item()

    return median


def transform_2d_coord_arrays(homography, in_coords):
    aug_coords = np.dstack([in_coords, np.ones(in_coords.shape[:2])]).squeeze()
    out_coordinates = np.einsum("ik,abk", homography, aug_coords)
    out_coordinates = (out_coordinates / out_coordinates[:, :, 2][:, :, None])[:, :, :2]

    return out_coordinates


def map_coordinates(pixel_indices, field, cval=0.0, mode="bilinear"):
    pixel_indices = pixel_indices.clone()
    yx_size = field.shape[-1], field.shape[-2]
    for i in range(2):
        pixel_indices[:, :, i] = ((pixel_indices[:, :, i]) / (yx_size[i] - 1) * 2) - 1

    out_of_bounds_mask = ((pixel_indices > 1) | (pixel_indices < -1)).any(dim=2)

    field = torch.nn.functional.grid_sample(
        field.unsqueeze(0).unsqueeze(0),
        pixel_indices.unsqueeze(0),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze()

    field[out_of_bounds_mask] = cval

    return field.squeeze()


def similarity_matrix(scale: float, angle: float, tvec: Iterable) -> np.ndarray:
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


def transform_img(
    img: torch.Tensor,
    transformation: np.ndarray,
    bgval: Optional[float] = None,
    mode: str = "bilinear",
    invert: bool = False,
) -> torch.Tensor:
    """
    Return translation vector to register images.

    Notes
    -----
    The transformation is to be understand on the coordinate axis of natural to an
    image array. That is, the origin of coordinates is on the top left with the y axis
    going down and the x axis to the right. Rotations go from x to y and therefore
    a positive rotation angle will result in an anticlockwise rotation from the point
    of view of the user when plotting the image but in fact its a clockwise rotation
    from the point of view of the coordinate system.

    Parameters
    ----------
    img
        What will be transformed.
        If a 3D array is passed, it is treated in a manner in which RGB
        images are supposed to be handled - i.e. assume that coordinates
        are (Y, X, channels).
        Complex images are handled in a way that treats separately
        the real and imaginary parts.
    mode
        `nearest` or `bilinear`. This are the modes supported by `grid_sample`
    bgval
        Shade of the background (filling during transformations)
        If None is passed, :func:`imreg_dft.utils.get_borderval` with
        radius of 5 is used to get it.

    Returns
    -------
    The transformed image

    """
    if invert:
        transformation = np.linalg.inv(transformation)

    if bgval is None:
        bgval = get_borderval(img)

    transformed_coords = transform_2d_coord_arrays(
        np.linalg.inv(transformation),
        np.dstack(
            np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]), indexing="xy")
        ),
    )
    transformed_coords = torch.tensor(transformed_coords).type(torch.float32)
    slave_transformed = map_coordinates(
        transformed_coords,
        img,
        cval=float(bgval),
        mode=mode,
    )

    dest = slave_transformed

    return dest
