import numpy as np
from dlt_homography import dlt_homography

if __name__ == "__main__":
    # Input point correspondences (4 points).
    I1pts = np.array([
        [0, 1, 1, 0],  # x-coordinates in Image 1
        [0, 0, 1, 1]   # y-coordinates in Image 1
    ])
    I2pts = np.array([
        [1, 3, 3, 1],  # x-coordinates in Image 2 (scaled and translated)
        [1, 1, 3, 3]   # y-coordinates in Image 2 (scaled and translated)
    ])

    H, A = dlt_homography(I1pts, I2pts)
    print(H)

# Don't forget: the homography operates on homogeneous points!