from pyathena.classic.cooling import coolftn
from pyathena.util.derived_fields import add_derived_fields
import xarray as xr
import pandas as pd
import numpy as np
import pickle

def dataset_tavg(s, ts, te):
    """ return time-averaged datasets """
    cf = coolftn()
    nums = np.arange(ts, te+1)
    sn0 = s.read_sn()[['time','x1sn','x2sn','x3sn']]
    dx = s.domain['dx'][0]
    dy = s.domain['dx'][1]
    dz = s.domain['dx'][2]
    zl = s.domain['center'][2] - 0.5*dz
    zr = s.domain['center'][2] + 0.5*dz
    tp = s.load_vtk(num=ts-1).domain['time']*s.u.Myr
    fields = ['density','velocity','pressure','gravitational_potential']
    time, Pdrive, Pturb = [], [], []

    # load a first vtk
    ds = s.load_vtk(num=ts)
    time.append(s.domain['time']*s.u.Myr)
    dat = ds.get_field(fields, as_xarray=True)
    add_derived_fields(s, dat, 'T')
    tmp = dat.where(dat.T < 2.0e4, other=1e-15) # two-phase medium
    dat = xr.concat([dat,tmp], pd.Index(['all', '2p'], name='phase'))
    dat = dat.drop('T')

    # Pdrive
    dt = time[-1]-tp
    sn = sn0[(sn0.time > tp)&(sn0.time < time[-1])]
    n0 = dat.density.sel(z=[zl,zr]).mean(dim=['z','y','x']).values
    # SNe in two-phase cloudy medium; eqn 34 in Kim & Ostriker (2015)
    pstar = 2.8e5 * n0**-0.17
    NSNe = len(sn)
    Pdrive.append(0.25*pstar*NSNe/dt)

    # Pturb
    pt = (dat.density*dat.velocity3**2).interp(z=0)
    pt = (pt.sum(dim=['y','x'])*dx*dy).values[()]*s.u.Msun*s.u.kms/s.u.Myr
    Pturb.append(pt)

    # loop through vtks
    for num in nums[1:]:
        ds = s.load_vtk(num=num)
        time.append(ds.domain['time']*s.u.Myr)
        tmp = ds.get_field(fields, as_xarray=True)
        add_derived_fields(s, tmp, 'T')
        tmp2 = tmp.where(tmp.T < 2.0e4, other=1e-15)
        tmp = xr.concat([tmp,tmp2], pd.Index(['all', '2p'], name='phase'))
        tmp = tmp.drop('T')
    
        # Pdrive
        dt = time[-1] - time[-2]
        sn = sn0[(sn0.time > time[-2])&(sn0.time < time[-1])]
        n0 = tmp.density.sel(z=[zl,zr]).mean(dim=['z','y','x']).values
        # SNe in two-phase cloudy medium; eqn 34 in Kim & Ostriker (2015)
        pstar = 2.8e5 * n0**-0.17
        NSNe = len(sn)
        Pdrive.append(0.25*pstar*NSNe/dt)
    
        # Pturb
        pt = (tmp.density*tmp.velocity3**2).interp(z=0)
        pt = (pt.sum(dim=['y','x'])*dx*dy).values[()]*s.u.Msun*s.u.kms/s.u.Myr
        Pturb.append(pt)
        
        # combine
        dat += tmp
    dat /= len(nums)
    dat['Pdrive'] = xr.DataArray(np.array(Pdrive, dtype=np.float32).T,
            dims=['phase','t'], coords={'phase':['all','2p'],'t':time})
    dat['Pturb'] = xr.DataArray(np.array(Pturb, dtype=np.float32).T,
            dims=['phase','t'], coords={'phase':['all','2p'],'t':time})
    return dat
