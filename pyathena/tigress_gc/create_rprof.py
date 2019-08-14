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

        datRz = dat.where(mask).mean(dim=['x','y']).expand_dims(dim={'R':Rbins[i:i+1]})
        datRz = datRz.assign(H=H.expand_dims('R'))
        datRz = datRz.assign(Hw=Hw.expand_dims('R'))
        datRz = datRz.assign(Hu=Hu.expand_dims('R'))
        datRz = datRz.assign(Hc=Hc.expand_dims('R'))

        newdat.append(datRz)

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

    # transform from (x,y,z) to (R,z) coordinates (azimuthal average)
    Rmin, Rmax, dR = 0, 256, 10
    Redges = np.arange(Rmin, Rmax, dR)
    Rbins = 0.5*(Redges[1:]+Redges[:-1])
    dat = _transform_to_Rz(dat, Redges)

    # add fields
    dat['Sigma_gas'] = ((dat['density']*ds.domain['dx'][axis_idx['z']]).sum(dim='z')*s.u.Msun/s.u.pc**2)
    dat['Pth'] = dat['pressure'].interp(z=0)
    dat['Pturb'] = (dat['density']*dat['velocity3']**2).interp(z=0)

    return Rbins, dat.drop(['density','velocity1','velocity2','velocity3','pressure'])
