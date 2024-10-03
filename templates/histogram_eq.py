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

    # Calculate the histogram of Image I intensities
    hist, bins = np.histogram(I, bins=256, range=(0, 256))

    # Compute the CDF
    cdf = hist.cumsum()

    # Normalize CDF to be in the range [0,1]
    cdf = cdf / cdf[-1]

    # Scale CDF to [0, 255] for intensity mapping
    cdf_scaled = cdf * 255

    # Round intensities to nearest integer
    cdf_scaled = np.round(cdf_scaled)

    # Apply the CDF mapping to the original image
    J = cdf_scaled[I]

    return J