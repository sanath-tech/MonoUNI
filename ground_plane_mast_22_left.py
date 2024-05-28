import numpy as np

# List of points
points_in_lidar = np.array([
    [-6.6442, 8.4829, 0.63475],
    [-6.5371, 8.557, 0.63432],
    [-6.4292, 8.6356, 0.63219],
    [-6.3563, 8.7687, 0.60742],
    [-9.2847, 9.0052, 0.7604],
    [-9.1274, 9.0725, 0.7759],
    [-9.1125, 9.166, 0.75896],
    [-9.096, 9.3688, 0.7188],
    [-11.849, 3.6054, 0.60986],
    [-11.737, 3.7312, 0.62973],
    [-11.71, 3.8019, 0.6303],
    [-11.57, 3.9168, 0.65821],
    [-14.731, 22.48, 0.98377],
    [-14.483, 22.698, 0.96969],
    [-21.362, 22.019, 1.2028],
    [-21.078, 22.264, 1.1964],
    [-26.742, 18.229, 1.17],
    [-9.4032, 17.73, 0.53094],
    [-26.697, 13.373, 1.1081],
    [-26.872, 12.849, 1.1255]
])
T_lidar_to_cam = np.array([[ 0.60433522,  0.79627156, -0.02702852 , 0.23113008],
 [ 0.15283375, -0.14915492, -0.97693124,  4.5667283 ],
 [-0.78193401,  0.58626308, -0.21183674, 0.76367209]])
points_in_lidar_hom = np.hstack((points_in_lidar, np.ones((points_in_lidar.shape[0], 1))))
points_in_camera_hom = points_in_lidar_hom @ T_lidar_to_cam.T
points = points_in_camera_hom[:, :3]
# Append a column of ones to the points
points_h = np.hstack((points, np.ones((points.shape[0], 1))))

# Perform SVD
U, S, Vt = np.linalg.svd(points_h)
V = Vt.T

# The plane parameters A, B, C, D are in the last column of V
plane_parameters = V[:, -1]

A, B, C, D = plane_parameters

print("Plane equation: {:.5f}x + {:.5f}y + {:.5f}z + {:.5f} = 0".format(A, B, C, D))
