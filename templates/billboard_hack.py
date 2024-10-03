# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite
import matplotlib.pyplot as plt

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """

    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences between images
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    # Read in the images
    Iyd = imread('images/yonge_dundas_square.jpg')
    Ist = imread('images/uoft_soldiers_tower_light.png')

    # Create numpy array for images
    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    # Perform histogram equalization on the sampling image
    J = histogram_eq(Ist)

    # Compute the DLT homography matrix
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Path object to define the area of the billboard in the Iyd image
    billboard_area = Path(Iyd_pts.T)

    # Iterate over the bounding box region to map pixels from Ist to Iyd
    for x in range(bbox[0].min(), bbox[0].max()):
        for y in range(bbox[1].min(), bbox[1].max()):
            # Check if the current point (x, y) is inside the billboard area
            if billboard_area.contains_point(np.array([[x], [y]])):
                # Create a coordinate for the point in Iyd
                pt = np.array([[x, y, 1]]).T.reshape(3,1)
                # Apply the homography transformation to map the point to Ist
                sampling_pt = H @ pt
                # Normalize to convert from homogeneous to Cartesian coordinates
                sampling_pt /= sampling_pt[2]
                # Extract the x and y coordinates
                sampling_pt = sampling_pt[:2]
                # Check if the sampling point is within the bounds of Ist
                if sampling_pt[0] >= 0 and sampling_pt[0] < Ist.shape[1] - 1:
                    if sampling_pt[1] >= 0 and sampling_pt[1] < Ist.shape[0] - 1:
                        #  Perform bilinear interpolation on J
                        Ihack[y, x] = bilinear_interp(J, sampling_pt)

    # # Visualize the result, if desired...
    # plt.imshow(Ihack)
    # plt.show()

    return Ihack

billboard_hack()