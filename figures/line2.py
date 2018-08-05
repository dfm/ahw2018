#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from plotting import setup, savefig, SQUARE_FIGSIZE

setup()
np.random.seed(42)

def log_prior(m, b, lns, x_true):
    if not -4.0 < m < 4.0:
        return -np.inf
    if not -4.0 < b < 4.0:
        return -np.inf
    if not -10.0 < lns < 5.0:
        return -np.inf
    if np.any(x_true < 0.0) or np.any(x_true > 10.0):
        return -np.inf
    return -1.5 * np.log(1 + b**2)

def log_likelihood(x, y, y_err, x_err, m, b, lns, x_true):
    mod = m * x_true + b
    var = np.sqrt(y_err**2 + np.exp(2*lns))
    return -0.5 * (
        np.sum((y-mod)**2/var + np.log(var)) +
        np.sum(((x-x_true)/x_err)**2)
    )

def log_posterior(theta, x, y, y_err, x_err):
    m, b, lns = theta[:3]
    x_true = theta[3:]
    return (
        log_prior(m, b, lns, x_true) +
        log_likelihood(x, y, y_err, x_err, m, b, lns, x_true)
    )

if __name__ == "__main__":
    import emcee
    import matplotlib.pyplot as plt

    x, y, y_err, x_err = np.loadtxt("../data.txt", unpack=True)

    A = np.vander(x, 2)
    ATA = np.dot(A.T, A / (y_err**2)[:, None])
    w = np.linalg.solve(ATA, np.dot(A.T, y / y_err**2))
    p0 = np.concatenate((w, [-5.0], x))

    # Run the sampler.
    ndim = len(p0)
    nwalkers = 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                    args=(x, y, y_err, x_err))
    pos, lp, _ = sampler.run_mcmc(p0+1e-4*np.random.randn(nwalkers, ndim),
                                  5000)
    ind = np.argmax(lp)
    sampler.reset()
    pos, lp, _ = sampler.run_mcmc(
        pos[ind]+1e-4*np.random.randn(nwalkers, ndim), 5000)
    sampler.run_mcmc(pos, 20000)

    fig, ax = plt.subplots(1, 1, figsize=SQUARE_FIGSIZE, sharex=True)
    ax.errorbar(x, y, yerr=y_err, xerr=x_err, fmt=".k", capsize=0, ms=4)
    x0 = np.linspace(0, 10, 3)
    samples = sampler.flatchain
    for i in np.random.randint(len(samples), size=100):
        theta = samples[i]
        ax.plot(x0, x0*theta[0] + theta[1], color="k", alpha=0.05)
    ax.plot(x0, 0.5*x0-0.1, label="truth")
    ax.plot(x0, np.dot(np.vander(x0, 2), w), label="naive fit")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.legend(fontsize=12, loc=4)
    savefig(fig, "line2.pdf")
