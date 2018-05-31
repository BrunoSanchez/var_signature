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


header = np.load('ATLAS_LC/features_header.npy')
header = np.concatenate([np.array(['OBJID']), header])

dmdt_idx = [i for i in range(len(header)) if 'DeltamDeltat' in header[i]]
sign_idx = [i for i in range(len(header)) if 'Signature' in header[i]]

tab = np.loadtxt('ATLAS_LC/features_c_MPULSE.txt.gz')

stack = np.average(tab, axis=0)
sign = stack[sign_idx]
plt.imshow(sign.reshape(18, 12).T, interpolation='none', cmap='Greys',
           vmax=2)
plt.title('MPULSE signature')
plt.savefig('mpulse_signature_c.png')


dmdt = stack[dmdt_idx]

plt.imshow(dmdt.reshape(23, 24).T, interpolation='none', cmap='rainbow')
plt.title('Mira dmdt')
plt.savefig('mpulse_dmdt_c.png')
plt.show()


#~ idxs = []
#~ unflat = np.zeros((18, 12))-2
#~ for idx, name in enumerate(header[sign_idx]):
    #~ phi_bin = int(name.split('_')[2])
    #~ mag_bin = int(name.split('_')[-1])

    #~ val = tab[10][sign_idx][idx]
    #~ unflat[phi_bin][mag_bin] = val
    #~ idxs.append([phi_bin, mag_bin])

#~ plt.imshow(unflat)
#~ plt.show()

#~ plt.subplot(161)
#~ plt.imshow(stack_sg.reshape(18, 12, order='F'))
#~ plt.subplot(162)
#~ plt.imshow(stack_sg.reshape(18, 12, order='A'))  #good
#~ plt.subplot(163)
#~ plt.imshow(stack_sg.reshape(18, 12, order='C')) #good
#~ plt.subplot(164)
#~ plt.imshow(stack_sg.reshape(12, 18, order='F').T) #good
#~ plt.subplot(165)
#~ plt.imshow(stack_sg.reshape(12, 18, order='C').T)
#~ plt.subplot(166)
#~ plt.imshow(stack_sg.reshape(12, 18, order='A').T)
#~ plt.show()

