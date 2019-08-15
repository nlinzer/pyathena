from .load_sim_tigress_gc import LoadSimTIGRESSGC
from ..util.units import Units
from ..io.read_starpar_vtk import read_starpar_vtk
import re
import pandas as pd
import numpy as np
import xarray as xr
from pyathena.classic.cooling import coolftn
from pyathena.util.wmean import wmean

Thot1, Twarm, Tcold = 2.e4, 5050, 184.

def create_Rzprof(s, num):
    """
    Transform dataset from (x,y,z) to (R,z).
    Scale height is a function of R only.
    """
    axis_idx = dict(x=0, y=1, z=2)
    # load vtk files
    ds = s.load_vtk(num=num)
    dat0 = ds.get_field(field=['density','velocity','pressure',
        'gravitational_potential'], as_xarray=True)

    # add additional fields
    cf = coolftn()
    u = Units()
    pok = dat0['pressure']*u.pok
    T1 = pok/(dat0['density']*u.muH)
    dat0['temperature'] = xr.DataArray(cf.get_temp(T1.values),
            coords=T1.coords, dims=T1.dims)
    zsq = xr.DataArray(dat0.z**2, name='zsq')
    dat0['zsq'] = xr.broadcast(zsq, dat0)[0]
    rcyl = xr.DataArray(np.sqrt(dat0.x**2+dat0.y**2), name='rcyl')
    dat0['rcyl'] = xr.broadcast(rcyl, dat0)[0]
    cos, sin = dat0.x/dat0.rcyl, dat0.y/dat0.rcyl
    dat0['vr'] = dat0.velocity1*cos + dat0.velocity2*sin
    dat0['vt'] = -dat0.velocity1*sin + dat0.velocity2*cos
    dat0['vz'] = dat0.velocity3
    dat0 = dat0.drop(['velocity1','velocity2','velocity3'])

    # phase labels
    warm = (dat0['temperature'] > Twarm)&(dat0['temperature'] < Thot1)
    unstable = (dat0['temperature'] > Tcold)&(dat0['temperature'] < Twarm)
    cold = dat0['temperature'] < Tcold
    twophase = dat0['temperature'] < Thot1

    # seperate different phases
    data = {'all':dat0,
            'w':dat0.where(warm),
            'u':dat0.where(unstable),
            'c':dat0.where(cold),
            '2p':dat0.where(twophase)}

    # Make radial bins
    Rmin, Rmax, dR = 0, 256, 10
    Redges = np.arange(Rmin, Rmax, dR)
    Rbins = 0.5*(Redges[1:]+Redges[:-1])
    Nr = len(Redges)-1

    newdata = {}
    for phase, dat in data.items():
        newdat = []
        for i in range(Nr):
            Rl, Rr = Redges[i], Redges[i+1]
            mask = (Rl < dat.rcyl)&(dat.rcyl < Rr)
            # calculate scale height at this annulus
            arr = dat.zsq.where(mask)
            H = np.sqrt(wmean(arr, dat['density'].where(arr.notnull()), dim=dat.dims))
            # average over an annulus
            datRz = dat.where(mask).mean(dim=['x','y']).expand_dims(dim={'R':Rbins[i:i+1]})
            datRz = datRz.assign(H=H.expand_dims('R'))
            newdat.append(datRz)
        newdat = xr.merge(newdat)
        newdata[phase] = newdat

    return newdata
