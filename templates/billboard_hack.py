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
    # Bounding box in Y & D Square image - use if you find useful.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('images/yonge_dundas_square.jpg')
    Ist = imread('images/uoft_soldiers_tower_light.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    # Histogram equalization first.
    J = histogram_eq(Ist)
    

    # Perspective Homography from Yonge Dundas to Soldiers Tower
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Create Path object to track shape of billboard to replace in image
    billboard_area = Path(Iyd_pts.T)

    # Map pixels in warped image back to pixel location in reference image
    for x in range(bbox[0].min(), bbox[0].max()):
        for y in range(bbox[1].min(), bbox[1].max()):
            if billboard_area.contains_point(np.array([[x], [y]])):
                # Find if point in path
                pt = np.array([[x, y, 1]]).T
                sampling_pt = H @ pt
                Ihack[y, x] = bilinear_interp(J, np.array([[sampling_pt[0][0]], [sampling_pt[1][0]]]))

    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    #------------------

    # # Visualize the result, if desired...
    plt.imshow(Ihack)
    plt.show()

    return Ihack

if __name__ == "__main__":
    billboard_hack()
