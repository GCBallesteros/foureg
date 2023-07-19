Foureg
======
Image registration using discrete Fourier transform.


Given two images, `foureg` calculates a similarity transformation that
transforms one image into the other.

Example
-------
The example transforms an image with a user defined transformation and then rediscovers
it using `foureg`.

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from foureg import (Constraints, frame_img, similarity, similarity_matrix,
                    transform_img)

# Generate the test images
transformation = similarity_matrix(0.8, -10, (60, 20))
master = np.asarray(Image.open("./resources/examples/sample1.png"))
master = torch.from_numpy(master.copy()).type(torch.float32)

slave = transform_img(master, transformation, invert=True)

# Define some constraints and coregister
constraints = Constraints(angle=(-10, 5), scale=(0.8, 0.2), tx=(60, 3), ty=(20, 1))
imreg_result = similarity(
    master, slave, constraints=constraints, numiter=5, filter_pcorr=5
)

# Transform the slave image
slave_transformed = transform_img(slave, imreg_result.transformation, invert=False)

_, axs = plt.subplots(1, 4, figsize=(13, 8))
im_0 = axs[0].imshow(master)
plt.colorbar(im_0, ax=axs[0])
im_1 = axs[1].imshow(slave)
plt.colorbar(im_1, ax=axs[1])
im_2 = axs[2].imshow(slave_transformed)
plt.colorbar(im_2, ax=axs[2])
im_3 = axs[3].imshow(np.abs(slave_transformed - master))
plt.colorbar(im_3, ax=axs[3])

plt.show()
```

Features
--------
* Image pre-processing options (frequency filtration, image extension).
* Under-the-hood options exposed (iterations, phase correlation filtration).
* Permissive open-source license (3-clause BSD).
* GPU accelerated

Origin story
------------
This is a fork of the [imreg_dft](https://github.com/matejak/imreg_dft) borned of the
desire to achieve the following goals:
- Ability to return the final transformation in matrix form as opposed to the angle,
translation and scaling factor separately. The original code makes obtaining that
matrix really hard because it does it performs using scipy in  away that each transformation
resizes the image.
- Better performance powered by pytorch
- A more focused codebase. The only goal here is to estimate similarity transformations
between pairs of images.


Acknowledgements
----------------
The code was originally developed by Christoph Gohlke (University of California, Irvine, USA)
and later on developed further by Matěj Týč (Brno University of Technology, CZ). This
repo wouldn't exist without them.
