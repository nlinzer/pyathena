# read_hst.py

import os
import numpy as np
import pandas as pd
from ..io.read_hst import read_hst

class ReadHst:

    def read_hst(self, savdir_hst=None, merge_mhd=True, force_override=False):
        """Function to read hst and convert quantities to convenient units

        """

        # Create savdir if it doesn't exist
        if savdir_hst is None:
            savdir_hst = os.path.join(self.savdir, 'hst')
        if not os.path.exists(savdir_hst):
            os.makedirs(savdir_hst)
            force_override = False

        fpkl = os.path.join(savdir_hst,
                            os.path.basename(self.files['hst']) + '.mod.p')

        # Check if the original history file is updated
        if not force_override and os.path.exists(fpkl) and \
           os.path.getmtime(fpkl) > os.path.getmtime(self.files['hst']):
            self.logger.info('[read_hst]: Reading from existing pickle.')
            hst = pd.read_pickle(fpkl)

        # If we are here, force_override is True or history file is updated.
        # Need to convert units and define new columns.
        u = self.u
        ds = self.ds
        # volume of resolution element
        dvol = ds.domain['dx'].prod()
        # total volume of domain
        vol = ds.domain['Lx'].prod()
        # Area of domain
        LxLy = ds.domain['Lx'][0]*ds.domain['Lx'][1]
        
        hst = read_hst(self.files['hst'], force_override=force_override)
        # delete the first row
        hst.drop(hst.index[:1], inplace=True)
        
        hst['time_code'] = hst['time']
        hst['time'] *= u.Myr # time in Myr
        hst['mass'] *= vol*u.Msun # total gas mass in Msun
        hst['Sigma_gas'] = hst['mass']/LxLy # Gas surface density in Msun/pc^2
        hst['scalar3'] *= vol*u.Msun # neutral gas mass in Msun 
        hst['Mion'] *= vol*u.Msun # (coll + ionrad) ionized gas mass in Msu
        hst['Mion_coll'] *= vol*u.Msun # (coll only before ray tracing) ionized gas mass in Msun
        hst['Qiphot'] *= vol*(u.length**3).cgs # photoionization rate in cgs units
        hst['Qicoll'] *= vol*(u.length**3).cgs # collisional ionization rate in cgs units
        hst['Qidust'] *= vol*(u.length**3).cgs # collisional ionization rate in cgs units

        # Mass fraction ionized gas
        hst['mf_ion'] = hst['Mion']/hst['mass']
        hst['mf_ion_coll'] = hst['Mion_coll']/hst['mass']

        for f in range(self.par['radps']['nfreq']):
            # Total luminosity [Lsun]
            hst['Ltot_cl{:d}'.format(f)] *= vol*u.Lsun
            hst['Ltot_ru{:d}'.format(f)] *= vol*u.Lsun
            hst['Ltot{:d}'.format(f)] = \
                hst['Ltot_cl{:d}'.format(f)] + hst['Ltot_ru{:d}'.format(f)]
            # Total luminosity included in simulation
            hst['L_cl{:d}'.format(f)] *= vol*u.Lsun
            hst['L_ru{:d}'.format(f)] *= vol*u.Lsun
            hst['L{:d}'.format(f)] = \
                hst['L_cl{:d}'.format(f)] + hst['L_ru{:d}'.format(f)]
            # Luminosity that escaped boundary
            hst['Lesc{:d}'.format(f)] *= vol*u.Lsun
            # Luminosity lost due to dmax
            hst['Llost{:d}'.format(f)] *= vol*u.Lsun
            # Escape fraction, lost fraction
            # Estimation of true escape fraction estimation (upper bound)
            hst['fesc{:d}'.format(f)] = hst['Lesc{:d}'.format(f)] / \
                                        hst['L{:d}'.format(f)]
            hst['flost{:d}'.format(f)] = hst['Llost{:d}'.format(f)] / \
                                         hst['L{:d}'.format(f)]
            hst['fesc{:d}_est'.format(f)] = hst['fesc{:d}'.format(f)] + \
                                            hst['flost{:d}'.format(f)]
            hst['fesc{:d}_cum_est'.format(f)] = \
                (hst['Lesc{:d}'.format(f)] + hst['Llost{:d}'.format(f)]).cumsum() / \
                 hst['L{:d}'.format(f)].cumsum()
                                                
        ##########################
        # With ionizing radiation
        ##########################
        if self.par['radps']['nfreq'] == 2 and \
           self.par['radps']['nfreq_ion'] == 1:
            hnu0 = self.par['radps']['hnu[0]']/u.eV
            hnu1 = self.par['radps']['hnu[1]']/u.eV
            # Total luminosity
            hst['Qitot_cl'] = hst['Ltot_cl0']/u.Lsun/hnu0/u.s
            hst['Qitot_ru'] = hst['Ltot_ru0']/u.Lsun/hnu0/u.s
            hst['Qitot'] = hst['Qitot_ru'] + hst['Qitot_cl']
            # Total Q included as source
            hst['Qi_cl'] = hst['L_cl0']/u.Lsun/hnu0/u.s
            hst['Qi_ru'] = hst['L_ru0']/u.Lsun/hnu0/u.s
            hst['Qi'] = hst['Qi_ru'] + hst['Qi_cl']
            hst['Qiesc'] = hst['Lesc0']/u.Lsun/hnu0/u.s
            hst['Qilost'] = hst['Llost0']/u.Lsun/hnu0/u.s
            hst['Qiesc_est'] = hst['Qilost'] + hst['Qiesc']

        else:
            self.logger.error('Unrecognized nfreq={0:d}, nfreq_ion={1:d}'.\
                              format(self.par['radps']['nfreq'],
                                     self.par['radps']['nfreq_ion']))

        # midplane radiation energy density in cgs units
        hst['Erad0_mid'] *= u.energy_density
        hst['Erad1_mid'] *= u.energy_density

        hst.index = hst['time_code']
        if merge_mhd:
            hst_mhd = self.read_hst_mhd()
            hst = hst_mhd.reindex(hst.index, method='nearest',
                                  tolerance=0.1).combine_first(hst)

        try:
            hst.to_pickle(fpkl)
        except IOError:
            self.logger.warinig('[read_hst]: Could not pickle hst to {0:s}.'.format(fpkl))

        return hst

    def read_hst_mhd(self):

        hst = read_hst('/tigress/changgoo/{0:s}/hst/{0:s}.hst'.\
                       format(self.problem_id))

        u = self.u
        domain = self.par['domain1']
        Lx = domain['x1max'] - domain['x1min']
        Ly = domain['x2max'] - domain['x2min']
        Lz = domain['x3max'] - domain['x3min']
        Nx = domain['Nx1']
        Ny = domain['Nx2']
        Nz = domain['Nx3']
        Ntot = Nx*Ny*Nz
        vol = Lx*Ly*Lz
        LxLy = Lx*Ly
        dz = Lz/Nz
        Omega = self.par['problem']['Omega']
        time_orb = 2*np.pi/Omega*u.Myr # Orbital time in Myr

        if 'x1Me' in hst:
            mhd = True
        else:
            mhd = False

        h = pd.DataFrame()
        h['time_code'] = hst['time']
        h['time'] = h['time_code']*u.Myr # time in Myr
        h['time_orb'] = h['time']/time_orb

        h['mass'] = hst['mass']*u.Msun*vol
        h['Sigma'] = h['mass']/LxLy
        h['mass_sp'] = hst['msp']*u.Msun*vol
        h['Sigma_sp'] = h['mass_sp']/LxLy

        # Mass, volume fraction, scale height
        for ph in ['c','u','w','h1','h2']:
            h['mf_{}'.format(ph)] = hst['M{}'.format(ph)]/hst['mass']
            h['vf_{}'.format(ph)] = hst['V{}'.format(ph)]
            h['H_{}'.format(ph)] = \
                np.sqrt(hst['H2{}'.format(ph)] / hst['M{}'.format(ph)])

        h['mf_2p'] = h['mf_c'] + h['mf_u'] + h['mf_w']
        h['vf_2p'] = h['vf_c'] + h['vf_u'] + h['vf_w']
        h['H_2p'] = np.sqrt((hst['H2c'] + hst['H2u'] + hst['H2w']) / \
                            (hst['Mc'] + hst['Mu'] + hst['Mw']))
        h['H'] = np.sqrt(hst['H2'] / hst['mass'])

        # Kinetic and magnetic energy
        h['KE'] = hst['x1KE'] + hst['x2KE'] + hst['x3KE']
        if mhd:
            h['ME'] = hst['x1ME'] + hst['x2ME'] + hst['x3ME']

        hst['x2KE'] = hst['x2dke']
        for ax in ('1','2','3'):
            Ekf = 'x{}KE'.format(ax)
            if ax == '2':
                Ekf = 'x2dke'
            # Mass weighted velocity dispersion??
            h['v{}'.format(ax)] = np.sqrt(2*hst[Ekf]/hst['mass'])
            if mhd:
                h['vA{}'.format(ax)] = \
                    np.sqrt(2*hst['x{}ME'.format(ax)]/hst['mass'])
            h['v{}_2p'.format(ax)] = \
                np.sqrt(2*hst['x{}KE_2p'.format(ax)]/hst['mass']/h['mf_2p'])
            
        h['cs'] = np.sqrt(hst['P']/hst['mass'])
        h['Pth_mid'] = hst['Pth']*u.pok
        h['Pth_mid_2p'] = hst['Pth_2p']*u.pok/hst['Vmid_2p']
        h['Pturb_mid'] = hst['Pturb']*u.pok
        h['Pturb_mid_2p'] = hst['Pturb_2p']*u.pok/hst['Vmid_2p']

        h['nmid'] = hst['nmid']
        h['nmid_2p'] = hst['nmid_2p']/hst['Vmid_2p']

        h['sfr10']=hst['sfr10']
        h['sfr40']=hst['sfr40']
        h['sfr100']=hst['sfr100']

        h.index = h['time_code']
        
        return h
    
        # return pd.read_pickle(
        #     '/tigress/changgoo/{0:s}/hst/{0:s}.hst_cal.p'.format(self.problem_id))