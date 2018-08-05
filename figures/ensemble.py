#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from traces import log_p_func
from plotting import setup, savefig, SQUARE_FIGSIZE

setup()
np.random.seed(42)

if __name__ == "__main__":
    import emcee
    from emcee import autocorr
    import matplotlib.pyplot as plt

    # Run the sampler.
    ndim = 2
    nwalkers = 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_p_func)
    sampler.run_mcmc(np.random.randn(nwalkers, ndim), 20000)
    tau = sampler.get_autocorr_time()
    acc_frac = np.mean(sampler.acceptance_fraction)

    with open("numbers-emcee.tex", "w") as f:
        f.write("% Automatically generated - emcee\n")
        f.write("\\newcommand{{\\eaccfrac}}{{{0:.2f}}}\n".format(acc_frac))
        f.write("\\newcommand{{\\etaua}}{{{0:.0f}}}\n".format(tau[0]))
        f.write("\\newcommand{{\\etaub}}{{{0:.0f}}}\n".format(tau[1]))

    fig, ax = plt.subplots(1, 1, figsize=SQUARE_FIGSIZE, sharex=True)
    p = autocorr.function(np.mean(sampler.chain, axis=0))
    ax.plot(p[:, 0], label=r"$f(\theta) = \theta_1$")
    ax.plot(p[:, 1], label=r"$f(\theta) = \theta_2$")
    p = autocorr.function(np.mean(np.prod(sampler.chain, axis=-1), axis=0))
    ax.plot(p, label=r"$f(\theta) = \theta_1 \, \theta_2$")
    ax.set_title("emcee")
    ax.set_ylabel("autocorrelation")
    ax.set_xlabel("lag [steps]")
    ax.set_xlim(0, 500)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.legend(fontsize=12)
    savefig(fig, "ensemble.pdf")
