Foureg
======
Image registration using discrete Fourier transform.


Given two images, `foureg` calculates a similarity transformation that
transforms one image into the other.

NOTE
----
THIS IS STILL WIP AND INTERFACES MAY CHANGE WITHOU NOTICE

Example
-------
The example transforms an image with a user defined transformation and then rediscovers
it using `foureg`.
```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Not a dependency from this pa

from foureg import similarity, similarity_matrix, transform_img

# 1) Make up some transformation
transformation = similarity_matrix(1.2, 15, (40, 60))

# 2) Open the master image and transform it to generate the slave image
master = np.asarray(Image.open("./resources/examples/sample1.png"))
slave = transform_img(master, transformation)


# 3) Use foureg to recover the transformation
imreg_result = similarity(master, slave)
slave_transformed = transform_img(slave, imreg_result["transformation"])

4) Some plotting to verify everything is working
_, axs = plt.subplots(1, 5, figsize=(13, 8))
im_0 = axs[0].imshow(master)
plt.colorbar(im_0, ax=axs[0])
im_1 = axs[1].imshow(slave)
plt.colorbar(im_1, ax=axs[1])
im_2 = axs[2].imshow(slave_transformed)
plt.colorbar(im_2, ax=axs[2])
im_3 = axs[3].imshow(imreg_result["timg"])
plt.colorbar(im_3, ax=axs[3])
im_4 = axs[4].imshow(np.abs(imreg_result["timg"] - master))
plt.colorbar(im_4, ax=axs[4])

plt.show()
```

Features
--------
* Image pre-processing options (frequency filtration, image extension).
* Under-the-hood options exposed (iterations, phase correlation filtration).
* Permissive open-source license (3-clause BSD).

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


Acknowledgements
----------------
The code was originally developed by Christoph Gohlke (University of California, Irvine, USA)
and later on developed further by Matěj Týč (Brno University of Technology, CZ). This
repo wouldn't exist without them.
