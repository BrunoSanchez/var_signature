#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  feets_test.py
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

import os
import numpy as np
import pandas as pd
import feets
import feets.preprocess
from astropy.io import fits

data_path = os.path.abspath('ATLAS_LC/')

def calc_features(varname):
    obj_tab = fits.getdata(os.path.join(data_path, varname+'_objects_bos0109.fit'))

    det_tab = fits.getdata(os.path.join(data_path, varname+'_detection_bos0109.fit'))

    fspace = feets.FeatureSpace(data=['time', 'magnitude', 'error'])
                        #    'magnitude2', 'time2', 'error2'])
    colnames = fspace.features_as_array_
    features_o = np.zeros(colnames.shape)
    features_c = np.zeros(colnames.shape)
    #for irow in range(len(obj_tab[0:3])):
        #anobj = obj_tab[irow]
    for anobj in obj_tab:
        objid = anobj['objid']
        detections = det_tab[det_tab['objid']==objid]
        o_tab = detections[detections['filter']=='o']
        c_tab = detections[detections['filter']=='c']

        #=========================================================================
        otime, omag, oerror = feets.preprocess.remove_noise(
                      o_tab['mjd'], o_tab['m'], o_tab['dm'])
        #print 'total cmag points', len(cmag)
        if len(otime)>=20:
            lc = [otime, omag, oerror]
            try:
                feat, val = fspace.extract(*lc)
            except:
                import ipdb; ipdb.set_trace()

            features_o = np.vstack([features_o, val])

        #=========================================================================
        ctime, cmag, cerror = feets.preprocess.remove_noise(
                      c_tab['mjd'], c_tab['m'], c_tab['dm'])
        #print 'total omag points', len(omag)
            if len(ctime)>=20:
                lc = [ctime, cmag, cerror]
                feat, val = fspace.extract(*lc)

                features_c = np.vstack([features_c, val])


    features_o = np.delete(features_o, 0, 0)
    features_c = np.delete(features_c, 0, 0)

    np.savetxt(os.path.join(data_path,'features_o_'+varname+'.txt.gz'), features_o)
    np.savetxt(os.path.join(data_path,'features_c_'+varname+'.txt.gz'), features_c)


if __name__=='__main__':
       import sys
       varname = sys.argv[-1]
       calc_features(varname)


