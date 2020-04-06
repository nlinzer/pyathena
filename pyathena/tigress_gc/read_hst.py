# read_hst.py

import os
import numpy as np
import pandas as pd
from astropy import units as au
from scipy import integrate

from ..io.read_hst import read_hst
from ..load_sim import LoadSim


class ReadHst:

    @LoadSim.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units
        """
        u = self.u
        domain = self.domain

        # volume of resolution element (code unit)
        dvol = domain['dx'].prod()
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()
        # domain length (code unit)
        Lx = domain['Lx'][0]
        Ly = domain['Lx'][1]
        Lz = domain['Lx'][2]

        hst = read_hst(self.files['hst'], force_override=force_override)

        # Total gas mass in Msun
        hst['mass'] *= vol
        hst['Mh2'] *= vol
        hst['Mh1'] *= vol
        hst['Mw'] *= vol
        hst['Mu'] *= vol
        hst['Mc'] *= vol
        hst['msp'] *= vol
        hst['msp_left'] *= vol

        # flux
        hst['F1_2p'] = hst['F1w']+hst['F1u']+hst['F1c']
        hst['F1'] = hst['F1h2']+hst['F1h1']+hst['F1_2p']

        hst['F2_2p'] = hst['F2w']+hst['F2u']+hst['F2c']
        hst['F2'] = hst['F2h2']+hst['F2h1']+hst['F2_2p']

        hst['F3_2p'] = hst['F3w']+hst['F3u']+hst['F3c']
        hst['F3'] = hst['F3h2']+hst['F3h1']+hst['F3_2p']

        # Total outflow mass
        hst['mass_out1'] = integrate.cumtrapz(hst['F1'], hst['time'], initial=0.0)*Ly*Lz
        hst['mass_out2'] = integrate.cumtrapz(hst['F2'], hst['time'], initial=0.0)*Lz*Lx
        hst['mass_out3'] = integrate.cumtrapz(hst['F3'], hst['time'], initial=0.0)*Lx*Ly

        # Calculate (cumulative) SN ejecta mass
        # JKIM: only from clustered type II(?)
        try:
            sn = read_hst(self.files['sn'], force_override=force_override)
            t_ = np.array(hst['time'])
            Nsn, snbin = np.histogram(sn.time, bins=np.concatenate(([t_[0]], t_)))
            hst['mass_snej'] = Nsn.cumsum()*self.par['feedback']['MejII'] # Mass of SN ejecta [Msun]
        except:
            raise ValueError("cannot read SN dump")

        # star formation rates [Msun/yr]
        hst['sfr1'] = hst['sfr1']*(Lx*Ly/1e6)
        hst['sfr5'] = hst['sfr5']*(Lx*Ly/1e6)
        hst['sfr10'] = hst['sfr10']*(Lx*Ly/1e6)
        hst['sfr40'] = hst['sfr40']*(Lx*Ly/1e6)
        hst['sfr100'] = hst['sfr100']*(Lx*Ly/1e6)

        self.hst = hst

        return hst

    def read_sn(self, savdir=None, force_override=False):
        """Load sn dump"""
        h = read_hst(self.files['sn'], force_override=force_override)
        self.sn = h
        return h
