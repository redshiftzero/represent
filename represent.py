#!/usr/bin/env python
import numpy as np
import pyfits
import pdb
import pandas as pd
import matplotlib.pyplot as plt

from mpltools import style
style.use('ggplot')


"""
Simple routine for visualizing how representative a training set is of
its testing set. This is useful for supervised machine learning methods
that rely on having a representative training set for accurate model
prediction. Each routine takes two column names and a two pandas
dataframes containing the training and testing data.

Example:

import represent
represent.hists("Elevation", "Rainfall", train, test)
"""


def hists(var1, var2, train, test):

    """
    Plot normalized training/testing sets versus var1 and var2
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    ax1.set_title('Testing Set')
    ax1.set_xlabel(var1)
    ax1.set_ylabel(var2)
    ax1.set_frame_on(True)

    ax2.set_title('Training Set')
    ax2.set_xlabel(var1)
    ax2.set_ylabel(var2)
    ax2.set_frame_on(True)

    min1 = np.min(np.array([np.min(test[var1]), np.min(train[var1])]))
    max1 = np.max(np.array([np.max(test[var1]), np.max(train[var1])]))
    min2 = np.min(np.array([np.min(test[var2]), np.min(train[var2])]))
    max2 = np.max(np.array([np.max(test[var2]), np.max(train[var2])]))

    H1, xedges1, yedges1, img1 = plt.hist2d(test[var1], test[var2],
    bins=100, range=np.array([(min1, max1), (min2, max2)]),
    cmap=plt.cm.jet, normed=True)
    extent = [yedges1[0], yedges1[-1], xedges1[0], xedges1[-1]]
    H2, xedges2, yedges2, img2 = plt.hist2d(train[var1], train[var2],
    bins=100, range=np.array([(min1, max1), (min2, max2)]),
    cmap=plt.cm.jet, normed=True)

    colormax = np.max(np.array([np.max(H1), np.max(H2)]))
    im1 = ax1.imshow(np.rot90(H1), cmap=plt.cm.jet, extent=extent,
                     vmax=colormax)
    im2 = ax2.imshow(np.rot90(H2), extent=extent, cmap=plt.cm.jet,
                     vmax=colormax)

    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im1, ax=ax1)
    plt.show()
    plt.savefig(var1 + '_v_' + var2 + '.png')
    plt.close('all')


def ratio(var1, var2, train, test):

    """
    Plot ratio of histograms of train to test sets
    """

    G1, xedges1, yedges1, img1 = plt.hist2d(test[var1], test[var2],
    bins=100, cmap=plt.cm.jet)
    G2, xedges2, yedges2, img2 = plt.hist2d(train[var1], train[var2],
    bins=100, cmap=plt.cm.jet)
    G = G2 / G1
    plt.close('all')

    fig, ax = plt.subplots(1, 1)
    extent = [yedges1[0], yedges1[-1], xedges1[0], xedges1[-1]]
    im1 = ax.imshow(np.rot90(G), cmap=plt.cm.jet, extent=extent)
    fig.colorbar(im1, ax=ax)
    ax.set_title('Ratio of Train/Test Densities')
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_frame_on(True)
    plt.show()
    plt.savefig(var1 + '_v_' + var2 + '_ratio.png')
