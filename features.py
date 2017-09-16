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

import csv
import os
import warnings

import numpy as np

import pandas as pd

import feets

from libs.mppandas import mp_apply
from libs.ext_signature import Signature


# =============================================================================
# CLASSES
# =============================================================================

class Extractor(object):

    def __init__(self, fs):
        self._fs = fs

    def __call__(self, df):
        fs = self._fs
        df[fs.features_as_array_] = df.LINEARobjectID.apply(self.extract)
        return df

    def extract(self, lc_id):
        print("Extracting {}...".format(lc_id))
        fs = self._fs
        lc_path = os.path.join('allDAT', "{}.dat".format(lc_id))
        lc = np.loadtxt(lc_path)
        time, mag, mag_err = lc[:,0], lc[:,1], lc[:,2]

        sort_mask = time.argsort()
        data = (mag[sort_mask], time[sort_mask], mag_err[sort_mask])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = dict(zip(*fs.extract_one(data)))

        return pd.Series(result)


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_catalog(path):
    with open(path) as fp:
        columns = fp.readlines()[32].split()[1:]

    arr = np.loadtxt(path)
    df = pd.DataFrame(arr, columns=columns)

    df["LINEARobjectID"] = df.LINEARobjectID.astype(int)
    df["LCtype"] = df.LCtype.astype(int)
    df["CUF"] = df.CUF.astype(int)
    df["t2"] = df.t2.astype(int)
    df["t3"] = df.t3.astype(int)

    return df


def main():
    feets.register_extractor(Signature)

    fs = feets.FeatureSpace(
        data=["magnitude", "time", "error"],
        exclude=["SlottedA_length",
                 "StetsonK_AC",
                 "StructureFunction_index_21",
                 "StructureFunction_index_31",
                 "StructureFunction_index_32"])
    extractor = Extractor(fs)

    df = get_catalog('catalogs/PLV_LINEAR.dat.txt')
    #~ extractor(df)

    features = mp_apply(df, extractor)
    features.to_pickle("features/features.pkl")
    features.to_csv("features/features.csv", index=False)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    main()
