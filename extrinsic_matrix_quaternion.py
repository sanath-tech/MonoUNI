import numpy as np
from scipy.spatial.transform import Rotation as R

# Given projection matrix P
P = np.array([
    [0.60433522, 0.79627156, -0.02702852, 0.23113008],
    [0.15283375, -0.14915492, -0.97693124, 4.5667283],
    [-0.78193401, 0.58626308, -0.21183674, 0.76367209]
])

# Extract rotation matrix R and translation vector t
R_matrix = P[:, :3]
t_vector = P[:, 3]

# Convert rotation matrix to quaternion
rot = R.from_matrix(R_matrix)
quaternion = rot.as_quat()  # This gives [x, y, z, w]

print("Rotation Matrix R:\n", R_matrix)
print("Translation Vector t:\n", t_vector)
print("Quaternion (x, y, z, w):\n", quaternion)
