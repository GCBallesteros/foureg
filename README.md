Foureg
======
Image registration using discrete Fourier transform.


Given two images, `foureg` calculates a similarity transformation the transforms that
transforms one into the other.

Origin story
------------
This is a fork of the [imreg_dft](https://github.com/matejak/imreg_dft) borned of the
desire to achieve the following goals:
- Ability to return the final transformation in matrix form as opposed to the angle,
translation and scaling factor separately. The original code makes obtaining that
matrix really hard because it does some unorthodox resizings when performing the
image transformations.
- Better performance and ultimately a Pytorch powered GPU implementation
- A more focused codebase. The only goal here is to estimate similarity transformations
between pairs of images.

Features
--------
* Image pre-processing options (frequency filtration, image extension).
* Under-the-hood options exposed (iterations, phase correlation filtration).
* Permissive open-source license (3-clause BSD).

Acknowledgements
----------------
The code was originally developed by Christoph Gohlke (University of California, Irvine, USA)
and later on developed further by Matěj Týč (Brno University of Technology, CZ). This
repo wouldn't exist without them.
