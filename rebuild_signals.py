#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rebuild_signals.py
#
#  Copyright 2018 Bruno S <bruno@oac.unc.edu.ar>
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

import numpy as np
import matplotlib.pyplot as plt


tab = np.loadtxt('ATLAS_LC/features_c_CBH.txt.gz')

header = np.load('ATLAS_LC/features_header.npy')

tab = tab[0:100]

dmdt_idx = [i for i in range(len(header)) if 'DeltamDeltat' in header[i]]

dmdt_obj0 =  tab[0][dmdt_idx]

plt.imshow(dmdt_obj0.reshape(24,23))
plt.show()

sign_idx = [i for i in range(len(header)) if 'Signature' in header[i]]

idxs = []
for idx, name in enumerate(header[sign_idx]):
    phi_bin = int(name.split('_')[2])
    mag_bin = int(name.split('_')[-1])

    idxs.append([phi_bin, mag_bin])

#~ sign_obj0 = tab[12][sign_idx]
#~ sign = np.zeros((12,18))
#~ for idx, val in enumerate(sign_obj0):
    #~ k, j = idxs[idx]
    #~ sign[j, k] = val

#~ plt.imshow(sign)
#~ plt.show()

stack = np.average(tab, axis=0)
sign = stack[sign_idx]
plt.imshow(sign.reshape(18, 12))
plt.show()


dmdt = stack[dmdt_idx]

plt.plot(dmdt)
plt.vlines(x=np.arange(0, 23*24, 23), ymax =256, ymin=0)
plt.show()


stack_sg = stack[sign_idx]
sign = np.zeros((18, 12))

for i in range(12):
    j = np.arange(i, 18*12, 12)
    print j
    sign[:, i] = stack_sg[j]


plt.subplot(161)
plt.imshow(stack_sg.reshape(18, 12, order='F'))
plt.subplot(162)
plt.imshow(stack_sg.reshape(18, 12, order='A'))
plt.subplot(163)
plt.imshow(stack_sg.reshape(18, 12, order='C'))
plt.subplot(164)
plt.imshow(stack_sg.reshape(12, 18, order='F').T)
plt.subplot(165)
plt.imshow(stack_sg.reshape(12, 18, order='C').T)
plt.subplot(166)
plt.imshow(stack_sg.reshape(12, 18, order='A').T)
plt.show()

plt.subplot(121)
plt.imshow(tab[0][sign_idx].reshape(18,12))
plt.subplot(122)
plt.imshow(s0.reshape(18,12))
plt.show()
