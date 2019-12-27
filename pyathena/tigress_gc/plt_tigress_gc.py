"""
*This module is not intended to be used as a script*
Putting a script inside a module's directory is considered as an antipattern
(see rejected PEP 3122).
You are encouraged to write a seperate script that executes the functions in
this module. - SMOON
"""
import os
import time
import os.path as osp
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr
from astropy import units as au
from astropy import constants as ac

from ..classic.cooling import coolftn
from ..io.read_hst import read_hst
from ..io.read_starpar_vtk import read_starpar_vtk
from ..util.units import Units

from pygc.cooling import Cooling
from pygc.util import add_derived_fields

def mass_norm(mass):
    '''
    Mass normlization function to determine symbol size
    This should be called both in sp_plot and sp_legend for the consistent result
    '''
    return np.sqrt(mass/300.)

def plt_all(s, num, fig, with_starpar=False, savfig=True):
    """
    Create large plot including density slice, density projection, temperature
    slice, phase diagram, star formation rate, and mass fractions.
    """
    # load vtk and hst files
    ds = s.load_vtk(num=num)
    dat = ds.get_field(field=['density','pressure'])
    hst = s.read_hst(force_override=True)
    time_code = ds.domain['time']
    time = time_code*s.u.Myr
    axis_idx = dict(x=0, y=1, z=2)
    idx = abs(hst.time-time_code)==abs(hst.time-time_code).min()
    heat_ratio = hst.loc[idx].heat_ratio
    column_density = (hst.loc[idx].mass.values*s.u.mass/
        (s.domain['Lx'][0]*s.domain['Lx'][1]*au.pc**2)/(s.u.muH*ac.m_p)).cgs.value
    efftau_in = s.par['problem']['efftau']
    c = Cooling(hr=heat_ratio, dx=s.domain['dx'][0], crNH=column_density, efftau=efftau_in)

    dmin = 1e-2
    dmax = 1e3
    sxymin = 1e0
    sxymax = 1e3
    sxzmin = 1e-2
    sxzmax = 1e4
    if "C0" in s.basename:
        Mdot = 0.125
        sfrlim = [1e-2,1e0]
        masslim = [1e5,1e7]
    elif "C1" in s.basename:
        Mdot = 0.5
        sfrlim = [1e-2,1e0]
        masslim = [1e6,1e8]
    elif "C2" in s.basename:
        Mdot = 2
        sfrlim = [1e-1,1e1]
        masslim = [1e6,1e8]
    elif "N0" in s.basename:
        Mdot = 0.125
        sfrlim = [1e-2,1e0]
        masslim = [1e6,1e8]
    elif "N1" in s.basename:
        Mdot = 0.5
        sfrlim = [1e-2,1e0]
        masslim = [1e6,1e8]
    elif "N2" in s.basename:
        Mdot = 2
        sfrlim = [1e-1,1e1]
        masslim = [1e7,1e9]
#    elif "V10_" in s.basename:
#        Mdot = 0.55 + 0.45*np.cos(2*np.pi*hst['time']*s.u.Myr/10)
#        sfrlim = [5e-2,5e0]
#        masslim = [5e5,5e7]
    else:
        raise Exception("set appropriate yranges for the model {}".format(s.basename))

    # create axes
    gs = GridSpec(3,5,figure=fig)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[2,0])
    ax3 = fig.add_subplot(gs[2,1])
    ax4 = fig.add_subplot(gs[0,2])
    ax5 = fig.add_subplot(gs[1,2])
    ax6 = fig.add_subplot(gs[2,2])
    ax7 = fig.add_subplot(gs[0,3])               # n-P phase diagram
    ax8 = fig.add_subplot(gs[0,4])               # n-T phase diagram
    ax9 = fig.add_subplot(gs[1,3:])              # SFR history
    ax10 = fig.add_subplot(gs[2,3:], sharex=ax9) # mass history
    cax1 = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05)
    cax2 = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05)
    cax3 = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05)
    cax4 = make_axes_locatable(ax4).append_axes('right', size='5%', pad=0.05)
    cax5 = make_axes_locatable(ax5).append_axes('right', size='5%', pad=0.05)
    cax6 = make_axes_locatable(ax6).append_axes('right', size='5%', pad=0.05)

    # prepare variables to be plotted
    dat['pok'] = dat['pressure']*s.u.pok
    add_derived_fields(dat, 'T')
    dat['surf_xy'] = ((dat['density']*ds.domain['dx'][axis_idx['z']])
            .sum(dim='z')*s.u.Msun/s.u.pc**2)
    dat['surf_xz'] = ((dat['density']*ds.domain['dx'][axis_idx['z']])
            .sum(dim='y')*s.u.Msun/s.u.pc**2)

    # plot

    # gas
    (dat['density'].interp(z=0)).plot.imshow(ax=ax4, norm=LogNorm(),
            cmap='viridis', vmin=dmin, vmax=dmax, cbar_ax=cax4, add_labels=False,
            cbar_kwargs={'label':r"$n_{\rm H}\,[\rm cm^{-3}]$"})
    (dat['density'].interp(y=0)).plot.imshow(ax=ax5, norm=LogNorm(),
            cmap='viridis', vmin=dmin, vmax=dmax, cbar_ax=cax5, add_labels=False,
            cbar_kwargs={'label':r"$n_{\rm H}\,[\rm cm^{-3}]$"})
    dat['surf_xy'].plot.imshow(ax=ax1, norm=LogNorm(), cmap='pink_r',
            vmin=sxymin, vmax=sxymax, cbar_ax=cax1, add_labels=False,
            cbar_kwargs={'label':r"$\Sigma_{\rm gas}\,[M_\odot\,\rm pc^{-2}]$"})
    dat['surf_xz'].plot.imshow(ax=ax6, norm=LogNorm(), cmap='pink_r',
            vmin=sxzmin, vmax=sxzmax, cbar_ax=cax6, add_labels=False,
            cbar_kwargs={'label':r"$\Sigma_{\rm gas}\,[M_\odot\,\rm pc^{-2}]$"})
    (dat['T'].interp(z=0)).plot.imshow(ax=ax2, norm=LogNorm(),
            cmap='coolwarm', vmin=1e1, vmax=1e8, cbar_ax=cax2, add_labels=False,
            cbar_kwargs={'label':r"$T\,[\rm K]$"})
    (dat['T'].interp(y=0)).plot.imshow(ax=ax3, norm=LogNorm(),
            cmap='coolwarm', vmin=1e1, vmax=1e8, cbar_ax=cax3, add_labels=False,
            cbar_kwargs={'label':r"$T\,[\rm K]$"})

    if with_starpar:
        # starpar
        try:
            sp = s.load_starpar_vtk(num=num)
            young_sp = sp[sp['age']*s.u.Myr < 40.]
            young_cluster = young_sp[young_sp['mass'] != 0]
            mass = young_cluster['mass']*s.u.Msun
            age = young_cluster['age']*s.u.Myr
            cl = ax1.scatter(young_cluster['x1'], young_cluster['x2'], marker='o',
                             s=mass_norm(mass), c=age, edgecolor='black', linewidth=1,
                             vmax=40, vmin=0, cmap=plt.cm.cool_r, zorder=2)
            ss=[]
            label=[]
            ext = ax1.images[0].get_extent()
            for mass in [1e4,1e5,1e6]:
                ss.append(ax1.scatter(ext[1]*2,ext[3]*2,s=mass_norm(mass),color='k',alpha=.5))
                label.append(r'$10^%d M_\odot$' % np.log10(mass))
            ax1.set_xlim(ext[0], ext[1])
            ax1.set_ylim(ext[3], ext[2])
            ax1.legend(ss,label,scatterpoints=1,loc=2,ncol=3,bbox_to_anchor=(0.0, 1.1), frameon=False)

            cax = fig.add_axes([0.15,0.93,0.25,0.015])
            cbar = plt.colorbar(cl, ticks=[0,20,40], cax=cax, orientation='horizontal')
            cbar.ax.set_title(r'$age\,[\rm Myr]$')
        except:
            # TODO this should catch proper exception
            pass

    # phase diagram
    nHlim = np.array([1e-1, 1e4])
    Plim = np.array([1e3, 1e7])
    Tlim = np.array([1e1, 1e5])

    w = dat.density.data.flatten()
    x = np.log10(w)
    y = np.log10(dat.pok.data.flatten())
    myrange = [np.log10(nHlim), np.log10(Plim)]
    pdf, xedge, yedge = np.histogram2d(x, y, bins=100, range=myrange, weights=w)
    ax7.pcolormesh(10**xedge, 10**yedge, pdf.T, norm=LogNorm(), cmap='plasma')
    ax7.set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
    ax7.set_ylabel(r'$P/k_{\rm B}\,[{\rm K\,cm^{-3}}]$')

    y = np.log10(dat.T.data.flatten())
    myrange = [np.log10(nHlim), np.log10(Tlim)]
    pdf, xedge, yedge = np.histogram2d(x, y, bins=100, range=myrange, weights=w)
    ax8.pcolormesh(10**xedge, 10**yedge, pdf.T, norm=LogNorm(), cmap='plasma')
    ax8.set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
    ax8.set_ylabel(r'$T\,[{\rm K}]$')

    # overplot LP threshold
    T = np.logspace(np.log10(12.95), 5)
    dx = s.domain['dx'][0]
    nth = c.get_rhoLPeq(dx, T)
    ax8.plot(nth, T, 'r--')
    prs = c.get_prs(nth, T)
    ax7.plot(nth, prs, 'r--')

    # overplot equilibrium curve
    nH = np.logspace(-1,4, 100)
    Teq = np.zeros(len(nH))
    for i in range(len(nH)):
        Teq[i] = c.get_Teq(nH[i], fuvle=True, cr=True)
    Peq = c.get_prs(nH, Teq)
    ax7.plot(nH, Peq, 'k--')
    ax8.plot(nH, Teq, 'k--')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax8.set_xscale('log')
    ax8.set_yscale('log')
    ax7.set_xlim(nHlim)
    ax7.set_ylim(Plim)
    ax8.set_xlim(nHlim)
    ax8.set_ylim(Tlim)

    # history
    ax9.semilogy(hst['time']*s.u.Myr, hst['sfr1'], 'm-', label='sfr1')
    ax9.semilogy(hst['time']*s.u.Myr, hst['sfr10'], 'r-', label='sfr10')
    ax9.semilogy(hst['time']*s.u.Myr, hst['sfr40'], 'g-', label='sfr40')
    ax9.semilogy(hst['time']*s.u.Myr, Mdot*np.ones(len(hst['time']*s.u.Myr))
        , 'k--', label='inflow')
    ax9.set_xlabel("time"+r"$\,[{\rm Myr}]$")
    ax9.set_ylabel("SFR"+r"$\,[M_\odot\,{\rm yr}^{-1}]$")
    ax9.set_ylim(sfrlim)
    ax9.plot([time,time],sfrlim,'y-',lw=5)
    ax9.legend()
    ax10.semilogy(hst['time']*s.u.Myr, hst['Mc']*s.u.Msun, 'b-', label=r"$M_c$")
    ax10.semilogy(hst['time']*s.u.Myr, hst['Mu']*s.u.Msun, 'g-', label=r"$M_u$")
    ax10.semilogy(hst['time']*s.u.Myr, hst['Mw']*s.u.Msun, 'r-', label=r"$M_w$")
    ax10.semilogy(hst['time']*s.u.Myr, hst['mass']*s.u.Msun, 'k-', label=r"$M_{\rm tot}$")
    ax10.semilogy(hst['time']*s.u.Myr, hst['msp']*s.u.Msun, 'k--', label=r"$M_{\rm sp}$")
    ax10.set_xlabel("time"+r"$\,[{\rm Myr}]$")
    ax10.set_ylabel("mass"+r"$\,[M_\odot]$")
    ax10.set_ylim(masslim)
    ax10.plot([time,time],masslim,'y-',lw=5)
    ax10.legend()

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set_aspect('equal')
    
    # figure annotations
    fig.suptitle('{0:s}, time: {1:.1f} Myr'.format(s.basename, time), fontsize=30, x=.5, y=.93)
    
    plt.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.1)

    if savfig:
        savdir = osp.join('./figures-all', s.basename)
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        fig.savefig(osp.join(savdir, 'all.{0:s}.{1:04d}.png'
            .format(s.basename, ds.num)))

def plt_history(s, fig, savfig=False):
    """
    Create history plot.
    """
    # create axes
    gs = GridSpec(2,1,figure=fig,hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # load vtk and hst files
    hst = s.read_hst(force_override=True)
    axis_idx = dict(x=0, y=1, z=2)

    if s.basename=="M0.1_2pc":
        Mdot = 0.1
        sfrlim = [1e-2,1e0]
        masslim = [1e5,1e7]
    elif s.basename=="M1_2pc":
        Mdot = 1
        sfrlim = [1e-1,1e1]
        masslim = [1e6,1e8]
    elif s.basename=="M10_2pc":
        Mdot = 10
        sfrlim = [1e0,1e2]
        masslim = [1e6,1e8]
    else:
        raise Exception("set appropriate ranges for the model {}".format(s.basename))

    # plot

    # history
    ax1.semilogy(hst['time']*s.u.Myr, hst['sfr1'], 'r-', label='sfr1')
    ax1.semilogy(hst['time']*s.u.Myr, hst['sfr5'], 'g-', label='sfr5')
    ax1.semilogy(hst['time']*s.u.Myr, hst['sfr10'], 'b-', label='sfr10')
    ax1.semilogy(hst['time']*s.u.Myr, hst['sfr40'], 'm-', label='sfr40')
    ax1.set_ylabel("SFR ["+r"$M_\odot\,{\rm yr}^{-1}$"+"]")
    ax1.set_ylim(sfrlim)
    ax1.legend()
    ax2.semilogy(hst['time']*s.u.Myr, hst['Mc']*s.u.Msun, 'b-', label=r"$M_c$")
    ax2.semilogy(hst['time']*s.u.Myr, hst['Mu']*s.u.Msun, 'g-', label=r"$M_u$")
    ax2.semilogy(hst['time']*s.u.Myr, hst['Mw']*s.u.Msun, 'r-', label=r"$M_w$")
    ax2.semilogy(hst['time']*s.u.Myr, hst['mass']*s.u.Msun, 'k-', label=r"$M_{\rm tot}$")
    ax2.semilogy(hst['time']*s.u.Myr, hst['msp']*s.u.Msun, 'k--', label=r"$M_{\rm sp}$")
    ax2.set_ylabel("mass ["+r"${M_\odot}$"+"]")
    ax2.set_ylim(masslim)
    ax2.legend()

    # figure annotations
    fig.suptitle('{0:s}'.format(s.basename), fontsize=30, x=.5, y=.93)
    
    if savfig:
        savdir = osp.join('./figures-all')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        fig.savefig(osp.join(savdir, 'history.{0:s}.png'
            .format(s.basename)),bbox_inches='tight')
