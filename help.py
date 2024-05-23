import yaml
from pyquaternion import Quaternion
import numpy as np


def read_kitti_ext(extfile):
    """read extrin"""
    text_file = open(extfile, 'r')
    cont = text_file.read()
    x = yaml.safe_load(cont)
    r = x['transform']['rotation']
    t = x['transform']['translation']
    q = Quaternion(r['w'], r['x'], r['y'], r['z'])
    m = q.rotation_matrix
    m = np.matrix(m).reshape((3, 3))
    t = np.matrix([t['x'], t['y'], t['z']]).T
    p1 = np.vstack((np.hstack((m, t)), np.array([0, 0, 0, 1])))
    return np.array(p1.I)