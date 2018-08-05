#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from plotting import setup, savefig, SQUARE_FIGSIZE

setup()
np.random.seed(42)

true_params = [0.5, -0.1, -0.3]

x_true = np.sort(np.random.uniform(0, 10, 8))
y_true = true_params[0] * x_true + true_params[1]

y_err = np.random.uniform(0.1, 0.2, len(y_true))
std = np.sqrt(y_err**2+np.exp(true_params[2]))
y = y_true + std*np.random.randn(len(y_true))

x_err = np.random.uniform(0.1, 0.2, len(x_true))
x = x_true + x_err*np.random.randn(len(x_true))

A = np.vander(x, 2)
ATA = np.dot(A.T, A / (y_err**2)[:, None])
w = np.linalg.solve(ATA, np.dot(A.T, y / y_err**2))

np.savetxt("../data.txt", np.vstack((x, y, y_err, x_err)).T)

fig, ax = plt.subplots(1, 1, figsize=SQUARE_FIGSIZE, sharex=True)
ax.errorbar(x, y, yerr=y_err, xerr=x_err, fmt=".k", capsize=0, ms=4)
x0 = np.linspace(0, 10, 3)
ax.plot(x0, np.dot(np.vander(x0, 2), true_params[:2]), label="truth")
ax.plot(x0, np.dot(np.vander(x0, 2), w), label="naive fit")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
plt.legend(fontsize=12, loc=4)
savefig(fig, "data.pdf")
