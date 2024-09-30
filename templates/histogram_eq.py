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

    # Create the CDF
    # Sorted Intensities 
    intensities = I.flatten()

    # Calculate the histogram
    hist, bins = np.histogram(intensities, bins=256)

    # CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]

    # Map Intensity values from Old Image to CDF
    hist_equalization_intensities = np.interp(intensities, bins[:,-1], cdf_normalized)
    J = hist_equalization_intensities.reshape(I.shape)


    return J