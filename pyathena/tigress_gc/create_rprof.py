from .load_sim_tigress_gc import LoadSimTIGRESSGC
from ..util.units import Units
from ..io.read_starpar_vtk import read_starpar_vtk
import re
import pandas as pd
import numpy as np
import xarray as xr
from pyathena.classic.cooling import coolftn
from pyathena.util.wmean import wmean

def _transform_to_Rz(dat, Redges):
    """
    Caveat: Scale heights need a special treatment here.
    """
    # update phase information
    cf = coolftn()
    u = Units()
    pok = dat['pressure']*u.pok
    T1 = pok/(dat['density']*u.muH)
    dat['temperature'] = xr.DataArray(cf.get_temp(T1.values), coords=T1.coords, dims=T1.dims)
    Thot1, Twarm, Tcold = 2.e4, 5050, 184.
    warm = (dat['temperature'] > Twarm)&(dat['temperature'] < Thot1)
    unstable = (dat['temperature'] > Tcold)&(dat['temperature'] < Twarm)
    cold = dat['temperature'] < Tcold

    Nr = len(Redges)-1
    dat['R'] = np.sqrt(dat.x**2+dat.y**2)
    newdat = []
    Rbins = 0.5*(Redges[1:]+Redges[:-1])
    for i in range(Nr):
        Rl, Rr = Redges[i], Redges[i+1]
        mask = (Rl < dat.R)&(dat.R < Rr)

        arr = (dat.z**2).where(mask)
        notnull = arr.notnull()
        H = np.sqrt(wmean(arr, dat['density'].where(notnull), dim=['x','y','z']))

        arr = (dat.z**2).where(warm&mask)
        notnull = arr.notnull()
        Hw = np.sqrt(wmean(arr, dat['density'].where(notnull), dim=['x','y','z']))

        arr = (dat.z**2).where(unstable&mask)
        notnull = arr.notnull()
        Hu = np.sqrt(wmean(arr, dat['density'].where(notnull), dim=['x','y','z']))

        arr = (dat.z**2).where(cold&mask)
        notnull = arr.notnull()
        Hc = np.sqrt(wmean(arr, dat['density'].where(notnull), dim=['x','y','z']))

        dat = dat.where(mask).mean(dim=['x','y']).expand_dims(dim={'R':Rbins[i:i+1]})
        dat = dat.assign(H=H.expand_dims('R'))
        dat = dat.assign(Hw=Hw.expand_dims('R'))
        dat = dat.assign(Hu=Hu.expand_dims('R'))
        dat = dat.assign(Hc=Hc.expand_dims('R'))

        newdat.append(dat.where(mask).mean(dim=['x','y']).expand_dims(dim={'R':Rbins[i:i+1]}))

    newdat = xr.merge(newdat)
    return newdat

def create_rprof(s, num):
    """
    Create radial profile
    """
    axis_idx = dict(x=0, y=1, z=2)
    # load vtk files
    ds = s.load_vtk(num=num)
    dat = ds.get_field(field=['density','velocity','pressure'], as_xarray=True)

#    # update phase information
#    cf = coolftn()
#    pok = dat['pressure']*s.u.pok
#    T1 = pok/(dat['density']*s.u.muH)
#    dat['temperature'] = xr.DataArray(cf.get_temp(T1.values), coords=T1.coords, dims=T1.dims)
#    Thot1, Twarm, Tcold = 2.e4, 5050, 184.
#    warm = (dat['temperature'] > Twarm)&(dat['temperature'] < Thot1)
#    unstable = (dat['temperature'] > Tcold)&(dat['temperature'] < Twarm)
#    cold = dat['temperature'] < Tcold
#    dat['z2w'] = (dat.z**2).where(warm)
#    dat['z2u'] = (dat.z**2).where(unstable)
#    dat['z2c'] = (dat.z**2).where(cold)

    # transform from (x,y,z) to (R,z) coordinates (azimuthal average)
    Rmin, Rmax, dR = 0, 256, 10
    Redges = np.arange(Rmin, Rmax, dR)
    Rbins = 0.5*(Redges[1:]+Redges[:-1])
    dat = _transform_to_Rz(dat, Redges)

    # add fields
    dat['Sigma_gas'] = ((dat['density']*ds.domain['dx'][axis_idx['z']]).sum(dim='z')*s.u.Msun/s.u.pc**2)
    dat['Pth'] = dat['pressure'].interp(z=0)
    dat['Pturb'] = (dat['density']*dat['velocity3']**2).interp(z=0)
#    dat['H'] = np.sqrt(wmean(dat.z**2, dat['density'], dim=['z']))
#    dat['Hw'] = np.sqrt(wmean(dat['z2w'], dat['density'], dim=['z']))
#    dat['Hu'] = np.sqrt(wmean(dat['z2u'], dat['density'], dim=['z']))
#    dat['Hc'] = np.sqrt(wmean(dat['z2c'], dat['density'], dim=['z']))

    return Rbins, dat.drop(['density','velocity1','velocity2','velocity3','pressure'])


#def _rprof(quantity, x, y, redges):
#    r = np.sqrt(x**2+y**2)
#    Nr = len(redges)-1
#    prof = np.zeros(Nr)
#    for i in range(Nr):
#        rl, rr = redges[i], redges[i+1]
#        mask = (rl < r)&(r < rr)
#        prof[i] = quantity.where(mask, drop=True).mean()
#    return prof

#    # load vtk and hst files
#    ds = s.load_vtk(num=num)
#    dat = ds.get_field(field=['density','velocity','pressure'], as_xarray=True)
#    sp = read_starpar_vtk(s.files['starpar'][num])
#    time = ds.domain['time']*s.u.Myr
#    axis_idx = dict(x=0, y=1, z=2)
#
#    # prepare variables to be plotted
#    dat['Sigma_gas'] = ((dat['density']*ds.domain['dx'][axis_idx['z']])
#                        .sum(dim='z')*s.u.Msun/s.u.pc**2)
#    dat['Pth'] = dat['pressure'].interp(z=0)
#    dat['Pturb'] = (dat['density']*dat['velocity3']**2).interp(z=0)
#    r = np.sqrt(dat.x**2+dat.y**2)
#    cos = dat.x/r
#    sin = dat.y/r
#    dat['vrot'] = (-dat['velocity1']*sin + dat['velocity2']*cos).interp(z=0)
#    dat['H'] = np.sqrt(wmean(dat.z**2, dat['density'], dim=['z']))
#    cf = coolftn()
#    pok = dat['pressure']*s.u.pok
#    T1 = pok/(dat['density']*s.u.muH)
#    dat['temperature'] = xr.DataArray(cf.get_temp(T1.values), coords=T1.coords, dims=T1.dims)
#    Thot1, Twarm, Tcold = 2.e4, 5050, 184.
#    warm = (dat['temperature'] > Twarm)&(dat['temperature'] < Thot1)
#    unstable = (dat['temperature'] > Tcold)&(dat['temperature'] < Twarm)
#    cold = dat['temperature'] < Tcold
#
#    dat['Hw'] = np.sqrt(wmean((dat.z**2).where(warm), dat['density'], dim=['z']))
#    dat['Hu'] = np.sqrt(wmean((dat.z**2).where(unstable), dat['density'], dim=['z']))
#    dat['Hc'] = np.sqrt(wmean((dat.z**2).where(cold), dat['density'], dim=['z']))
#
#
#
#    quantities = ['Sigma_gas', 'Pth', 'Pturb', 'vrot', 'H', 'Hw', 'Hu', 'Hc']
#
#    # radial binning
#    rmin, rmax, dr = 0, 256, 10
#    redges = np.arange(rmin, rmax, dr)
#    rprof = {}
#    for quantity in quantities:
#        rprof[quantity] = _rprof(dat[quantity], dat.x, dat.y, redges)
#    rbins = 0.5*(redges[1:]+redges[:-1])
#    return rbins, rprof

#def _get_sfr(time, tbin):
#    u = Units()
#    ageMyr = (time - sf_data['time']*u.Myr)
#    mask = (0 < ageMyr)&(ageMyr < tbin)
#    sfr = sf_data[mask]['mstar'].sum()*u.Msun / tbin / 1e6
#    return sfr # Msun / yr


#def _parse_line(rx, line):
#    """
#    Do a regex search against given regex and
#    return the match result.
#
#    """
#
#    match = rx.search(line)
#    if match:
#        return match
#    # if there are no matches
#    return None
#
#def parse_file(filepath):
#    """
#    Parse text at given filepath
#
#    Parameters
#    ----------
#    filepath : str
#        Filepath for file_object to be parsed
#
#    Returns
#    -------
#    data : pd.DataFrame
#        Parsed data
#
#    """
#
#    data = []  # create an empty list to collect the data
#    # open the file and read through it line by line
#    with open(filepath, 'r') as file_object:
#        line = file_object.readline()
#        while line:
#            # at each line check for a match with a regex
#            rx = re.compile(r't=([0-9\.]+).*x=\((-?\d+),(-?\d+),(-?\d+)\).*n=([0-9\.]+).*nth=([0-9\.]+).*P=([0-9\.]+).*cs=([0-9\.]+)')
#            match = _parse_line(rx, line)
#            if match:
#                time = float(match[1])
#                x = float(match[2])
#                y = float(match[3])
#                z = float(match[4])
#                rho = float(match[5])
#                rho_crit = float(match[6])
#                prs = float(match[7])
#                cs = float(match[8])
#                line = file_object.readline()
#                rx = re.compile(r'navg=(-?[0-9\.]+).*id=(\d+).*m=(-?[0-9\.]+).*nGstars=(\d+)')
#                match = _parse_line(rx, line)
#                navg = float(match[1])
#                idstar = int(match[2])
#                mstar = float(match[3])
#                nGstars = int(match[4])
#                if (mstar > 0):
#                    row = {
#                        'time': time,
#                        'x': x,
#                        'y': y,
#                        'z': z,
#                        'rho': rho,
#                        'rho_crit': rho_crit,
#                        'prs': prs,
#                        'cs': cs,
#                        'navg': navg,
#                        'idstar': idstar,
#                        'mstar': mstar,
#                        'nGstars': nGstars
#                    }
#                    data.append(row)
#            else:
#                line = file_object.readline()
#
#        # create a pandas DataFrame from the list of dicts
#        data = pd.DataFrame(data)
#        # set the School, Grade, and Student number as the index
#        data.sort_values('time', inplace=True)
#    return data
#
#
##sf_data = parse_file("sf.txt")
