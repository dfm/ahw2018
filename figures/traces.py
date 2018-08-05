#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from plotting import setup, savefig, SQUARE_FIGSIZE

setup()
np.random.seed(42)

Sigma = np.array([[1.0, -0.08], [-0.08, 0.01]])
def log_p_func(theta):
    return -0.5 * np.dot(theta, np.linalg.solve(Sigma, theta))

def mh(log_p_func, theta0, niter, sigma=0.1):
    ndim = len(theta0)
    theta = np.array(theta0)
    chain = np.empty((niter, ndim))
    lp = log_p_func(theta0)
    acc = 0
    for i in range(niter):
        q = np.array(theta)
        ind = np.random.randint(ndim)
        q[ind] += sigma * np.random.randn()
        lq = log_p_func(q)

        u = np.log(np.random.rand())
        if u < lq - lp:
            theta = q
            lp = lq
            acc += 1

        chain[i] = theta

    return chain, acc / niter

if __name__ == "__main__":
    import corner
    from emcee import autocorr
    import matplotlib.pyplot as plt

    # Run the sampler.
    chain, acc_frac = mh(log_p_func, np.random.randn(2), 400000)
    tau = autocorr.integrated_time(chain)
    print("Acceptance fraction: {0:.3f}".format(acc_frac))
    print("Autocorrelation times: {0}, {1}"
          .format(*(map("{0:.0f}".format, tau))))
    with open("numbers-mh.tex", "w") as f:
        f.write("% Automatically generated\n")
        f.write("\\newcommand{{\\accfrac}}{{{0:.2f}}}\n".format(acc_frac))
        f.write("\\newcommand{{\\taua}}{{{0:.0f}}}\n".format(tau[0]))
        f.write("\\newcommand{{\\taub}}{{{0:.0f}}}\n".format(tau[1]))

    # Plot the traces and corner plot.
    fig, axes = plt.subplots(2, 1, figsize=SQUARE_FIGSIZE, sharex=True)
    axes[0].plot(chain[:5000, 0], "k")
    axes[1].plot(chain[:5000, 1], "k")
    axes[0].set_ylabel(r"$\theta_1$")
    axes[1].set_ylabel(r"$\theta_2$")
    axes[1].set_xlabel("step")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(4))
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(3))
    savefig(fig, "traces.pdf")

    plt.close(fig)
    fig = corner.corner(chain, labels=[r"$\theta_1$", r"$\theta_2$"])
    savefig(fig, "corner.pdf")

    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=SQUARE_FIGSIZE, sharex=True)
    p = autocorr.function(chain)
    ax.plot(p[:, 0], label=r"$f(\theta) = \theta_1$")
    ax.plot(p[:, 1], label=r"$f(\theta) = \theta_2$")
    p = autocorr.function(np.prod(chain, axis=1))
    ax.plot(p, label=r"$f(\theta) = \theta_1 \, \theta_2$")
    ax.set_title("Metropolis")
    ax.set_ylabel("autocorrelation")
    ax.set_xlabel("lag [steps]")
    ax.set_xlim(0, 5000)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.legend(fontsize=12)
    savefig(fig, "autocorr.pdf")
