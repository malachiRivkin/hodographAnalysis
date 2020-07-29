# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:23:39 2020

@author: Malachi
"""

import numpy as np

from math import atan2
from numpy.linalg import eig, inv, svd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


df = np.genfromtxt('C:/Users/Malachi/OneDrive - University of Idaho/%SummerInternship2020/hodographAnalysis/microHodographs/fname.csv', delimiter=',', skip_header=1, names=['alt', 'u', 'v', 'temp', 'bv2'])
print(df)

u = df['u']
v = df['v']


''' fit_ellipse.py by Nicky van Foreest '''

def ellipse_center(a):
    """@brief calculate ellipse centre point
    @param a the result of __fit_ellipse
    """
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_axis_length(a):
    """@brief calculate ellipse axes lengths
    @param a the result of __fit_ellipse
    """
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) *\ ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) *\
            ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        

    res1 = np.sqrt(abs(up / down1))     #Added abs() to bothlines
    res2 = np.sqrt(abs(up / down2))

    return np.array([res1, res2])


def ellipse_angle_of_rotation(a):
    """@brief calculate ellipse rotation angle
    @param a the result of __fit_ellipse
    """
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return atan2(2 * b, (a - c)) / 2

def fmod(x, y):
    """@brief floating point modulus
        e.g., fmod(theta, np.pi * 2) would keep an angle in [0, 2pi]
    @param x angle to restrict
    @param y end of  interval [0, y] to restrict to
    """
    r = x
    while(r < 0):
        r = r + y
    while(r > y):
        r = r - y
    return r


def __fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points
                (internal method.. use fit_ellipse below...)
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    return U[:, 0]


def fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points: the 5 params
        returned are:
        a - major axis length
        b - minor axis length
        cx - ellipse centre (x coord.)
        cy - ellipse centre (y coord.)
        phi - rotation angle of ellipse bounding box
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    e = __fit_ellipse(x, y)
    centre, phi = ellipse_center(e), ellipse_angle_of_rotation(e)
    axes = ellipse_axis_length(e)
    a, b = axes

    # assert that a is the major axis (otherwise swap and correct angle)
    if(b > a):
        tmp = b
        b = a
        a = tmp

        # ensure the angle is betwen 0 and 2*pi
        phi = fmod(phi, 2. * np.pi)   #originally alpha = ...
    return [a, b, centre[0], centre[1], phi]

#--------------

eps = fit_ellipse(u,v)
a = eps[0]
b = eps[1]
x = eps[2]
y = eps[3]
phi = eps[4]

alpha = np.linspace(0, 2 * np.pi)
X = a * np.cos(alpha) * np.cos(phi) - b * np.sin(alpha) * np.sin(phi) + x
Y = a * np.cos(alpha) * np.sin(phi) + b * np.sin(alpha) * np.cos(phi) + y

xy = eps[2], eps[3]

ax = plt.axes()
plt.plot(u,v)
plt.plot(X,Y)
print(eps[2], eps[3], eps[0])
#fitted = mpatches.Ellipse(xy,eps[1] * 2,eps[2] * 2, eps[4], fill=False)
#ax.add_patch(fitted)
#plt.show()
