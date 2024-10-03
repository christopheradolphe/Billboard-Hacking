import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    four pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # Interpolation Performed with 4 surrounding pixels
    # x1, y1 are floor of point and x2, y2 are 1 greater than the floor

    # Get pixel coordinates
    x = pt[0,0]
    y = pt[1,0]
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = x1 + 1
    y2 = y1 + 1

    # Get intensity values
    b11 = I[y1, x1]
    b12 = I[y1, x2]
    b21 = I[y2, x1]
    b22 = I[y2, x2]

    # Interpolation weights 
    wx = x - x1
    wy = y - y1

    # Bilinear Interpolation
    b1 = (1 - wx) * b11 + wx * b12
    b2 = (1 - wx) * b21 + wx * b22
    b = (1 - wy) * b1 + wy * b2


    return b