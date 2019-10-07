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
        # area of domain
        area = [Ly*Lz, Lz*Lx, Lx*Ly]

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
            h['mass_in'] = Mdot*h['time']
        else:
            #TODO time varying inflow
            pass

        # Total outflow mass
        h['mass_out1'] = 0
        h['mass_out2'] = 0
        h['mass_out3'] = 0
        for i, direction in enumerate(['F1','F2','F3']):
            flux = ((hst[direction+'_upper'] - hst[direction+'_lower']).to_numpy()
                    *area[i]*u.mass_flux*u.length**2).to("Msun/Myr").value
            h['flux'+str(i+1)] = flux
            h['mass_out'+str(i+1)] += integrate.cumtrapz(flux, h['time'], initial=0.0)
        h['mass_out2'] += h['mass_in']

        # Total outflow mass
        h['mass_out'] = 0
        for i, direction in enumerate(['F1','F2','F3']):
            flux = ((hst[direction+'_upper'] - hst[direction+'_lower']).to_numpy()
                    *area[i]*u.mass_flux*u.length**2).to("Msun/Myr").value
            h['mass_out'] += integrate.cumtrapz(flux, h['time'], initial=0.0)
        h['mass_out'] += h['mass_in']

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
        """Function to read sn dump and convert quantities to convenient units
        """

        u = self.u

        hst = read_hst(self.files['sn'], force_override=force_override)

        h = pd.DataFrame()

        h['id'] = hst['id']
        # Time in code unit
        h['time_code'] = hst['time']
        # Time in Myr
        h['time'] = hst['time']*u.Myr
        # starpar age
        h['age'] = hst['age']*u.Myr
        # mass-weighted starpar age
        h['mage'] = hst['mage']*u.Myr
        # starpar mass
        h['mass'] = hst['mass']*u.Msun
        # position
        h['x1'] = hst['x1']
        h['x2'] = hst['x2']
        h['x3'] = hst['x3']
        # sn position TODO Why they are same with x1,x2,x3?
        h['x1sn'] = hst['x1sn']
        h['x2sn'] = hst['x2sn']
        h['x3sn'] = hst['x3sn']
        # feedback mode
        h['mode'] = hst['mode']
        # fm_sedov = 0.1
        h['fm'] = hst['fm']

        self.sn = h

        return h
