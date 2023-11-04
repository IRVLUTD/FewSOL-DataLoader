# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

class MarkerBoard:
    def __init__(self):

        # all unit in centimeters
        self.width = 70.4
        self.height = 50.4
        self.center = np.array([35.2, 25.2])

        # markers, the coordinate origin is the upper-left corner
        self.num_markers = 18
        self.marker_size = 5.9
        self.markers = {}
        self.markers['marker_00'] = np.array([2.95, 2.95])
        self.markers['marker_01'] = np.array([15.85, 2.95])
        self.markers['marker_02'] = np.array([28.75, 2.95])
        self.markers['marker_03'] = np.array([41.55, 2.95])
        self.markers['marker_04'] = np.array([54.45, 2.95])
        self.markers['marker_05'] = np.array([67.45, 2.95])

        self.markers['marker_06'] = np.array([67.45, 13.75])
        self.markers['marker_07'] = np.array([67.45, 25.00])
        self.markers['marker_08'] = np.array([67.45, 36.20])
        self.markers['marker_09'] = np.array([67.45, 47.45])

        self.markers['marker_10'] = np.array([54.45, 47.45])
        self.markers['marker_11'] = np.array([41.55, 47.45])
        self.markers['marker_12'] = np.array([28.70, 47.45])
        self.markers['marker_13'] = np.array([15.80, 47.45])
        self.markers['marker_14'] = np.array([2.95, 47.45])

        self.markers['marker_15'] = np.array([2.95, 36.75])
        self.markers['marker_16'] = np.array([2.95, 25.55])
        self.markers['marker_17'] = np.array([2.95, 14.25])

    # get relative pose between marker and center
    def get_relative_pose(self, idx):
        key = 'marker_%02d' % (idx)
        location = self.markers[key].copy()
        location[1] *= -1
        center = self.center.copy()
        center[1] *= -1
        diff = (center - location) * 0.01

        RT = np.eye(4, dtype=np.float32)
        RT[0, 3] = diff[0]
        RT[1, 3] = diff[1]
        return RT


    def project_center(self, pose, intrinsic_matrix):
        vertices = np.array([[0, 0, 0, 1]])
        vertices = np.matmul(pose, vertices.transpose())
        x2d = np.matmul(intrinsic_matrix, vertices[:3, :])
        x2d[0, :] = x2d[0, :] / x2d[2, :]
        x2d[1, :] = x2d[1, :] / x2d[2, :]
        return x2d[:2, :].flatten()


    def project_inside_roi(self, pose, intrinsic_matrix):
        w = (self.width - 2 * self.marker_size) * 0.01
        h = (self.height - 2 * self.marker_size) * 0.01
        vertices = np.array([[-w/2, -h/2, 0, 1],
                             [w/2, -h/2, 0, 1],
                             [w/2, h/2, 0, 1],
                             [-w/2, h/2, 0, 1]])
        vertices = np.matmul(pose, vertices.transpose())
        x2d = np.matmul(intrinsic_matrix, vertices[:3, :])
        x2d[0, :] = x2d[0, :] / x2d[2, :]
        x2d[1, :] = x2d[1, :] / x2d[2, :]
        return x2d[:2, :]


marker_board = MarkerBoard()

# compute marker board center with RANSAC
def compute_marker_board_center(meta):

   # collect hypotheses
    keys = []
    RT_centers = []
    for key in meta:
        if 'ar_marker' in key:
            idx = int(key[-2:])
            if idx >= marker_board.num_markers:
                continue
            RT_relative = marker_board.get_relative_pose(idx)

            pose = meta[key].flatten()
            RT = np.eye(4, dtype=np.float32)
            RT[:3, :3] = quat2mat(pose[3:])
            RT[:3, 3] = pose[:3]

            RT_final = np.matmul(RT, RT_relative)
            keys.append(key)
            RT_centers.append(RT_final)

    # compute errors for hypotheses
    num = len(keys)
    errors = np.zeros((num, ), dtype=np.float32)
    angles = np.zeros((num, ), dtype=np.float32)
    for i in range(num):
        RT_center = RT_centers[i]
        error = 0
        angle = 0
        for j in range(num):
            if j == i:
                continue

            # pose from hypothesis
            RT_relative = marker_board.get_relative_pose(j)
            RT = np.matmul(RT_center, np.linalg.inv(RT_relative))

            # pose from observation
            pose = meta[keys[j]].flatten()
            RT_marker = np.eye(4, dtype=np.float32)
            RT_marker[:3, :3] = quat2mat(pose[3:])
            RT_marker[:3, 3] = pose[:3]

            # angular error
            q1 = mat2quat(RT_marker[:3, :3])
            q2 = mat2quat(RT[:3, :3])
            angle += 2 * np.arccos(np.dot(q1, q2))

            # translation error
            t1 = RT_marker[:3, 3]
            t2 = RT[:3, 3]
            distance = np.linalg.norm(t1 - t2)
            error += distance
        errors[i] = error
        angles[i] = angle

    # find the minimum error hypothesis
    if num > 0:
        index = np.argmin(errors)
        meta['center'] = RT_centers[index]
    return meta