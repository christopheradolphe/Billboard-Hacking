import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """

    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # Calculate the histogram
    hist, bins = np.histogram(I, bins=256)

    # CDF
    cdf = hist.cumsum() / I.size
    cdf_scaled = cdf * 255
    cdf_scaled = np.round(cdf_scaled)

    J = cdf_scaled[I]

    return J