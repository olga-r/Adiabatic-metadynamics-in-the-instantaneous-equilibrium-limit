import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from matplotlib.figure import Figure
import matplotlib as mpl
import sys
import os
import glob, re
from scipy.integrate import quad


def potential_poly(x, left, right):
    c = (left + right) / 2
    return 50*((x-c)**2-0.1)**2



def potential_flat(x, left, right, delta = 0.01):
    c = (left + right) / 2
    xL = left + delta
    xR = right - delta
    def V(x):
        return 50 * (((x - c)**2 - 0.1)**2)

    def dV(x):
        return 200 * (x - c) * (((x - c)**2 - 0.1))

    def sigma(t):
        if t <= 0:
            return 0.0
        elif t >= 1:
            return 1.0
        else:
            return 6*t**5 - 15*t**4 + 10*t**3



    def sL(x):
        return sigma((x - left) / delta)

    def sR(x):
        return sigma((right - x) / delta)

    VL = V(xL) - quad(lambda u: sL(u) * dV(u), left, xL)[0]
    VR = V(xR) + quad(lambda u: sR(u) * dV(u), xR, right)[0]

    def V_total(x):
        if x <= left:
            return VL
        elif x < xL:
            return V(xL) - quad(lambda u: sL(u) * dV(u), x, xL)[0]
        elif x <= xR:
            return V(x)
        elif x < right:
            return V(xR) + quad(lambda u: sR(u) * dV(u), xR, x)[0]
        else:
            return VR

    vectorized =  np.vectorize(V_total)
    return vectorized(x)


def potential_exponential(x, left, right, alpha=100.0):
    c = (left + right) / 2

    V_edge = 50 * ((right - c)**2 - 0.1)**2
    dV_right = 200 * (right - c) * ((right - c)**2 - 0.1)
    dV_left = 200 * (left - c) * ((left - c)**2 - 0.1)

    A_right = dV_right / alpha
    B_right = V_edge - A_right

    A_left = -dV_left / alpha
    B_left = V_edge - A_left

    conditions = [
        x < left,
        (x >= left) & (x <= right),
        x > right
    ]

    choices = [
        A_left * np.exp(alpha * (left - x)) + B_left,
        50 * ((x - c)**2 - 0.1)**2,
        A_right * np.exp(alpha * (x - right)) + B_right
    ]

    return np.select(conditions, choices)



def potential_superexponential(x, left, right, beta=1.0):
    c = (left + right) / 2

    V_edge = 50 * ((right - c)**2 - 0.1)**2
    dV_right = 200 * (right - c) * ((right - c)**2 - 0.1)
    dV_left = 200 * (left - c) * ((left - c)**2 - 0.1)

    gamma_right = dV_right / V_edge
    gamma_left = -dV_left / V_edge

    conditions = [
        x < left,
        (x >= left) & (x <= right),
        x > right
    ]

    dx_left = left - x
    dx_right = x - right

    choices = [
        V_edge * np.exp(beta * dx_left**2 + gamma_left * dx_left),
        50 * ((x - c)**2 - 0.1)**2,
        V_edge * np.exp(beta * dx_right**2 + gamma_right * dx_right)
    ]

    return np.select(conditions, choices)

if __name__== "__main__":
    x = np.arange(-0.1, 1.101, 0.001)
    left = 0.0
    right = 1.0

    v1 = potential_flat(x, left, right)
    v2 = potential_exponential(x, left, right, alpha=20.0)
    v3 = potential_superexponential(x, left, right, beta=100)
    v4 = potential_poly(x, left, right)

    fig = Figure(linewidth=4, figsize = (5,5), dpi =300)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=1, left =0.12, right=0.85, bottom=0.17, top=0.92,  wspace=0.2, hspace=0.2)
    ##Запятой разделяются два числа или диапазона чисел. Первый - вовлеченные строки, второй - вовлеченные столбцы

    ax1 = fig.add_subplot(gs[0,0])

    ax1.plot(x, v1, lw=0.6, c="b")
    ax1.plot(x, v4, lw=0.6, c="r")
    ax1.plot(x, v2, lw=0.6, c="g")

    ax1.plot(x, v3, lw=0.6, c="m")


    fig.savefig(os.path.normpath('./potentials.png'))
