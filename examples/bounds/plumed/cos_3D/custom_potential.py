import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from matplotlib.figure import Figure
import matplotlib as mpl
import sys
import os
import glob, re
from scipy.integrate import quad
from scipy.special import ellipk

def custom(x, left, right):
    def poly(x, left, right):
        c = (left+right)/2
        return 250*((x-c)**2-0.13)**2
    return corrected_cos3d_potential(poly, x, left, right )




def omega3(s: float) -> float:
    """Geometric density for s=(3+cos x+cos y+cos z)/6."""
    if s <= 0.0 or s >= 1.0:
        return 0.0
    t = 6.0 * s - 3.0
    points = []
    if -1.0 < t < 1.0:
        points.append(np.arccos(t))
    for c in (t - 2.0, t + 2.0):
        if -1.0 < c < 1.0:
            points.append(np.arccos(c))
    points = sorted(set(points))

    def integrand(theta: float) -> float:
        d = t - np.cos(theta)
        if abs(d) > 2.0:
            return 0.0
        m = 1.0 - 0.25 * d * d
        m = min(max(m, 0.0), np.nextafter(1.0, 0.0))
        return ellipk(m)

    value, error = quad(
        integrand,
        0.0,
        np.pi,
        points=points,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )
    return 6.0 * value / np.pi**3




def corrected_cos3d_potential(func, x, left, right ):
    omega = np.array([omega3(si) for si in x])
    V_s = func(x, left, right)
    F_theory = V_s -  np.log(omega)
    F_theory -= np.nanmin(F_theory)
    return F_theory

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


    fig.savefig(os.path.normpath('./test.png'))
