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

        h = pd.DataFrame()

        # Time in code unit
        h['time_code'] = hst['time']
        # Time in Myr
        h['time'] = hst['time']*u.Myr

        # Total gas mass in Msun
        h['mass'] = hst['mass']*vol*u.Msun
        h['Mh2'] = hst['Mh2']*vol*u.Msun
        h['Mh1'] = hst['Mh1']*vol*u.Msun
        h['Mw'] = hst['Mw']*vol*u.Msun
        h['Mu'] = hst['Mu']*vol*u.Msun
        h['Mc'] = hst['Mc']*vol*u.Msun
        h['msp'] = hst['msp']*vol*u.Msun
        h['msp_left'] = hst['msp_left']*vol*u.Msun

        if self.par['problem']['iflw_flag']==1: #constant inflow rate
            # Total inflow mass
            Mdot = (2*self.par['problem']['iflw_d0']*self.par['problem']['iflw_v0']\
                     *self.par['problem']['iflw_mu']*self.par['problem']['iflw_w']\
                     *self.par['problem']['iflw_h']*self.u.density*self.u.velocity\
                     *self.u.length**2).to("Msun/Myr").value
            h['mdot_in'] = Mdot
            h['mass_in'] = Mdot*h['time']
        else:
            dT = self.par['problem']['iflw_dT']*self.u.Myr
            h['mass_in'] = 0.55*h['time'] + dT/(2*np.pi)*0.45*np.sin(2*np.pi*h['time']/dT)

        # flux
        h['F1h2'] = hst['F1h2']*u.Msun/u.Myr
        h['F1h1'] = hst['F1h1']*u.Msun/u.Myr
        h['F1w'] = hst['F1w']*u.Msun/u.Myr
        h['F1u'] = hst['F1u']*u.Msun/u.Myr
        h['F1c'] = hst['F1c']*u.Msun/u.Myr
        h['F1_2p'] = h['F1w']+h['F1u']+h['F1c']
        h['F1'] = h['F1h2']+h['F1h1']+h['F1_2p']

        h['F2h2'] = hst['F2h2']*u.Msun/u.Myr
        h['F2h1'] = hst['F2h1']*u.Msun/u.Myr
        h['F2w'] = hst['F2w']*u.Msun/u.Myr
        h['F2u'] = hst['F2u']*u.Msun/u.Myr
        h['F2c'] = hst['F2c']*u.Msun/u.Myr
        h['F2_2p'] = h['F2w']+h['F2u']+h['F2c']
        h['F2'] = h['F2h2']+h['F2h1']+h['F2_2p']

        h['F3h2'] = hst['F3h2']*u.Msun/u.Myr
        h['F3h1'] = hst['F3h1']*u.Msun/u.Myr
        h['F3w'] = hst['F3w']*u.Msun/u.Myr
        h['F3u'] = hst['F3u']*u.Msun/u.Myr
        h['F3c'] = hst['F3c']*u.Msun/u.Myr
        h['F3_2p'] = h['F3w']+h['F3u']+h['F3c']
        h['F3'] = h['F3h2']+h['F3h1']+h['F3_2p']

        # Total outflow mass
        h['mass_out1'] = integrate.cumtrapz(h['F1'], h['time'], initial=0.0)*Ly*Lz
        h['mass_out2'] = integrate.cumtrapz(h['F2'], h['time'], initial=0.0)*Lz*Lx
        h['mass_out3'] = integrate.cumtrapz(h['F3'], h['time'], initial=0.0)*Lx*Ly

#        # Calculate (cumulative) SN ejecta mass
#        # JKIM: only from clustered type II(?)
#        try:
#            sn = read_hst(self.files['sn'], force_override=force_override)
#            t_ = np.array(hst['time'])
#            Nsn, snbin = np.histogram(sn.time, bins=np.concatenate(([t_[0]], t_)))
#            h['mass_snej'] = Nsn.cumsum()*self.par['feedback']['MejII'] # Mass of SN ejecta [Msun]
#        except:
#            pass

        # star formation rates [Msun/yr]
        h['sfr1'] = hst['sfr1']*(Lx*Ly/1e6)
        h['sfr5'] = hst['sfr5']*(Lx*Ly/1e6)
        h['sfr10'] = hst['sfr10']*(Lx*Ly/1e6)
        h['sfr40'] = hst['sfr40']*(Lx*Ly/1e6)
        h['sfr100'] = hst['sfr100']*(Lx*Ly/1e6)

        self.hst = h

        return h

    def read_sn(self, savdir=None, force_override=False):
        """Load sn dump"""
        h = read_hst(self.files['sn'], force_override=force_override)
        self.sn = h
        return h
