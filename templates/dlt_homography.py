import numpy as np
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """

    # Create variables for point location coordinates
    x = I1pts[0, :]
    y = I1pts[1, :]
    u = I2pts[0, :]
    v = I2pts[1, :]

    # Create empty DLT Matrix
    A = np.empty((8, 9))

    # Populate the DLT Matrix
    # Add two equations to the DLT matrix for each of the four points
    for point in range(u.shape[0]):
        A[2 * point] = [-x[point], -y[point], -1,  0,    0,    0,   u[point]*x[point],  u[point]*y[point],  u[point]]
        A[2 * point + 1] = [ 0,     0,    0, -x[point], -y[point], -1,  v[point]*x[point],  v[point]*y[point],  v[point]]

    # Homography Matrix
    # H is the nullspace of matrix A
    null = null_space(A)
    H = null.reshape((3,3))

    # Normalize Matrix so bottom right entry is 1
    H = H / H[2,2]


    return H, A