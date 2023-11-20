# 
from pickletools import uint8
import numpy as np
from scipy import signal
from Video_Process.nailfold_video_process.utils.distance_geometry import get_distance,lookup_angle
import cv2

NEAR = 5
FAR = 50
ANGLE_RANGE = 10

def reconnect_skeleton(thinned, angle_field, debug=False):
    """将中断的骨架线重新链接

    Args:
        thinned (np.array): 骨架线的0/255二值图像
    """
    # 检验周边点个数
    kernel0 = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ])
    # 检验周边点是否连续
    kernel1 = np.array([
        [1,-2,3],
        [-8,0,-4],
        [7,-6,5]
    ])
    kernel2 = np.array([
        [7,-6,5],
        [-8,0,-4],
        [1,-2,3]
    ])
    binary_thinned = thinned.astype(np.bool8)
    border = signal.convolve2d(binary_thinned, kernel0, 'same')
    neighbor1 = signal.convolve2d(binary_thinned, kernel1, 'same')
    neighbor2 = signal.convolve2d(binary_thinned, kernel2, 'same')
    two_neighbors = (neighbor1 == 1) | (neighbor1 == -1) | (neighbor2 == 1) | (neighbor2 == -1)
    all = (two_neighbors & (border==2)) | (border<=1)
    all = all & binary_thinned
    if debug:
        point_map = all.astype(np.uint8)*255
        dilate_kernel = np.ones((3,3),np.uint8)
        point_map = cv2.dilate(point_map,dilate_kernel,iterations = 1)
        cv2.imwrite("points.png", point_map)
    points = np.where(all)
    points = [(points[0][i],points[1][i]) for i in range(points[0].shape[0])]
    for p in points:
        for q in points:
            if get_distance(p,q) < NEAR or get_distance(p,q) > FAR:
                continue
            direction = np.array(p) - np.array(q)
            angle = np.arctan( direction[0]/(direction[1]+1e-8) ) * (180 / np.pi)
            if angle < 0:
                angle += 180
            if abs(lookup_angle(p,angle_field) - angle) > ANGLE_RANGE:
                continue
            if abs(lookup_angle(q,angle_field) - angle) > ANGLE_RANGE:
                continue
            direction = np.sign(direction)
            if direction[0] != 0:
                if binary_thinned[q[0]+direction[0]][q[1]]:
                    continue
                if binary_thinned[p[0]-direction[0]][p[1]]:
                    continue
            if direction[1] != 0:
                if binary_thinned[q[0]][q[1]+direction[1]]:
                    continue
                if binary_thinned[p[0]][p[1]-direction[1]]:
                    continue
            if binary_thinned[q[0]+direction[0]][q[1]+direction[1]]:
                continue
            if binary_thinned[p[0]-direction[0]][p[1]-direction[1]]:
                continue
            
            cv2.line(thinned, (p[1],p[0]), (q[1],q[0]), 255, 1, 4)
    return thinned