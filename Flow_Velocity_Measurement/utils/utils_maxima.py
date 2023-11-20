import numpy as np
from scipy.signal import argrelextrema


def find_extrema(mtx, a=3):
    if len(mtx) == 0:
        return []
    w = mtx.shape[1]
    # w0 = int(87.2/180*w)
    # w1 = int(88.95/180*w)
    # a = round(w1-w0)+1
    # TODO: a and check the range
    mtx_prior = mtx[:, w//2-a*2+2:w//2+2]
    l0 = argrelextrema(mtx_prior, np.greater_equal, axis=0, order=a//2)
    l1 = argrelextrema(mtx_prior, np.greater_equal, axis=1, order=a//2)
    a0 = [p for p in zip(l0[0],l0[1])]
    a1 = [p for p in zip(l1[0],l1[1])]
    pts = list(set(a0) & set(a1))
    extremas = []

    pts_maxima = []
    mtx_prior0 = np.zeros(mtx_prior.shape)
    th = 0.5 # 0.3
    for pt in pts:
        if mtx_prior[pt[0]][pt[1]] >= np.max(np.max(mtx_prior[max(0,pt[0]-a//2):min(pt[0]+a//2+1,mtx.shape[0]), max(0,pt[1]-a):min(pt[1]+a+1, mtx.shape[1])])):
            if mtx_prior[pt[0]][pt[1]] >= th:
                pts_maxima.append((pt[0],pt[1]))
                mtx_prior0[pt[0]][pt[1]] = 1

    for pt in pts_maxima:
        if mtx_prior0[pt[0]][pt[1]] == 1:
            extremas.append((pt[0],pt[1]+w//2-a*2+2))
            mtx_prior0[max(0,pt[0]-a//2):min(pt[0]+a//2+1,mtx.shape[0]), max(0,pt[1]-a):min(pt[1]+a+1, mtx.shape[1])] = 0
    return extremas


def filter(sinogram, extremas):
    if len(sinogram) == 0:
        return []
    degrees = [extrema[1]/sinogram.shape[1]*180 for extrema in extremas]
    extremas_filter = []
    true_degrees = []
    for extrema,degree in zip(extremas,degrees):
        if 88.902 >= degree and degree >= 87.074:
            true_degrees.append(degree)
            extremas_filter.append(extrema)
    return extremas_filter


def vote(sinogram, extremas):
    if len(sinogram) == 0:
        return []
    extremas_refine = []
    a = 3
    for extrema in extremas:
        vote_square = sinogram[max(0,extrema[0]-a):min(sinogram.shape[0],extrema[0]+a+1),max(0,extrema[1]-a):min(sinogram.shape[1],extrema[1]+a+1)]
        sumup = np.sum(np.sum(vote_square))
        sum_x = np.sum(vote_square, axis = 0)
        sum_y = np.sum(vote_square, axis = 1)
        x = extrema[0]-a + np.dot(sum_x, np.array([i for i in range(len(sum_x))])) / sumup
        y = extrema[1]-a + np.dot(sum_y, np.array([i for i in range(len(sum_y))])) / sumup
        extremas_refine.append((x,y))
        # if x != extrema[0] or y != extrema[1]:
            # print(f"revised {extrema} to {(x,y)}")
    return extremas_refine
