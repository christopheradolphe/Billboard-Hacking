import numpy as np
from numpy.linalg import inv, norm
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
    u = I1pts[0, :]
    v = I1pts[1, :]
    x = I2pts[0, :]
    y = I2pts[1, :]
    A = np.array()

    for point in range(u.shape[1]):
        two_equations = np.array([
            [-x[point], -y[point], -1,  0,    0,    0,   u[point]*x[point],  u[point]*y[point],  u[point]],
            [ 0,     0,    0, -x[point], -y[point], -1,  v[point]*x[point],  v[point]*y[point],  v[point]]
        ])
        A = np.append(A, two_equations)
    
    A = np.array([
        # First point Correspondance
        [-x[0], -y[0], -1,  0,    0,    0,   u[0]*x[0],  u[0]*y[0],  u[0]],
        [ 0,     0,    0, -x[0], -y[0], -1,   v[0]*x[0],  v[0]*y[0],  v[0]]
        # Second Point Correspondance
        [],
        [],
        # Third Point Correspondance
        [],
        [],
        # Fourth Point Correspondance
        [],
        []

    ])

    return H, A