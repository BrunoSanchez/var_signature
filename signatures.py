#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  signatures.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii


def from_ij_to_idx(i, j, n_i):
    idx = j * n_i + i
    return idx

def from_idx_to_ij(idx, n_i):
    frac = float(idx)/float(n_i)
    j = int(frac)
    i = int((frac - j) * n_i)
    return i, j

def extract_signature(lc_time, lc_mag, lc_amplitude, lc_period, bins):
    lc_yaxis = (lc_mag - np.min(lc_mag) ) /np.float(lc_amplitude)

    # SHIFT TO BEGIN AT MINIMUM
    loc = np.argmin(lc_yaxis)
    lc_phase = np.remainder(lc_time - lc_time[loc], lc_period) / lc_period

    h = np.histogram2d(lc_phase, lc_yaxis, bins=bins, normed=True)

    return h


cat = np.loadtxt('PLV_LINEAR.dat.txt')


# =============================================================================
# Example i
# =============================================================================

def stack_var_class(i_class, cat):
    xbins = 30  # np.arange(0,1.1, 1/30.)
    ybins = 20  # np.arange(0, 1.5, 1/30.)

    h = np.zeros((30, 20)) # (len(ybins)-1, len(xbins)-1))

    for i in range(len(cat)):
        if cat[i][1] == i_class:
            lc_id = cat[i][0]
            lc_period = cat[i][2]
            lc_amplitude = cat[i][3]

            lc_path = os.path.join('allDAT', str(int(lc_id))+'.dat')

            lc_data = np.loadtxt(lc_path)
            lc_time = np.asarray(lc_data[:,0])

            h += extract_signature(lc_time,
                                   lc_data[:,1],
                                   lc_amplitude, lc_period,
                                   (xbins, ybins))[0]
    return h

rrlyrab = stack_var_class(1, cat)
rrlyrc = stack_var_class(2, cat)
algol1 = stack_var_class(3, cat)
algol2 = stack_var_class(4, cat)
contact_bin = stack_var_class(5, cat)
delta_scu = stack_var_class(6, cat)
lpvar = stack_var_class(7, cat)
hbeat = stack_var_class(8, cat)
bl_her = stack_var_class(9, cat)
anomal_cepheid = stack_var_class(11, cat)

# =============================================================================
# plot
# =============================================================================
plt.figure(figsize=(8, 16))

plt.subplot(5, 2, 1)
plt.imshow(rrlyrab.T, cmap='gray_r', interpolation='none')
plt.title('RR Lyrae ab')

plt.subplot(5, 2, 2)
plt.imshow(rrlyrc.T, cmap='gray_r', interpolation='none')
plt.title('RR Lyrae c')

plt.subplot(5, 2, 3)
plt.imshow(algol1.T, cmap='gray_r', interpolation='none')
plt.title('Algolyd 1 minimum')

plt.subplot(5, 2, 4)
plt.imshow(algol2.T, cmap='gray_r', interpolation='none')
plt.title('Algolyd 2 minimum')

plt.subplot(5, 2, 5)
plt.imshow(contact_bin.T, cmap='gray_r', interpolation='none')
plt.title('Contact Binaries')

plt.subplot(5, 2, 6)
plt.imshow(delta_scu.T, cmap='gray_r', interpolation='none')
plt.title('Delta scuti')

plt.subplot(5, 2, 7)
plt.imshow(lpvar.T, cmap='gray_r', interpolation='none')
plt.title('Long period var')

plt.subplot(5, 2, 8)
plt.imshow(hbeat.T, cmap='gray_r', interpolation='none')
plt.title('Heart Beat')

plt.subplot(5, 2, 9)
plt.imshow(bl_her.T, cmap='gray_r', interpolation='none')
plt.title('BL hear')

plt.subplot(5, 2, 10)
plt.imshow(anomal_cepheid.T, cmap='gray_r', interpolation='none')
plt.title('Anomal Cepheid')

plt.tight_layout()
plt.savefig('signature_stacking.png', dpi=600)

