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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr
from astropy import units as au

from ..classic.cooling import coolftn
from ..io.read_hst import read_hst
from ..io.read_starpar_vtk import read_starpar_vtk
from ..util.units import Units

def _get_histogram(s, num):
    # load vtk and hst files
    ds = s.load_vtk(num=num)
    dat = ds.get_field(field=['density','pressure'], as_xarray=True)
    time = ds.domain['time']*s.u.Myr
    axis_idx = dict(x=0, y=1, z=2)

    # prepare variables to be plotted
    dat['pok'] = dat['pressure']*s.u.pok
    # T_1 = (p/k) / (rho m_p) is the temperature assuming mu=1
    dat['T1'] = dat['pok']/(dat['density']*s.u.muH)
    dat['temperature'] = xr.DataArray(coolftn().get_temp(dat['T1'].values),
            coords=dat['T1'].coords, dims=dat['T1'].dims)

    # phase diagram
    hist_nP,xedge_nP,yedge_nP = np.histogram2d(
            np.log10(np.array(dat['density']).flatten()),
            np.log10(np.array(dat['pok']).flatten()), bins=200,
            range=[[-3,5],[2,8]],density=True,
            weights=np.array(dat['density']).flatten())
    hist_nT,xedge_nT,yedge_nT = np.histogram2d(
            np.log10(np.array(dat['density']).flatten()),
            np.log10(np.array(dat['temperature']).flatten()), bins=200,
            range=[[-3,5],[1,7]],density=True,
            weights=np.array(dat['density']).flatten())

    return hist_nP, xedge_nP, yedge_nP, hist_nT, xedge_nT, yedge_nT


def mass_norm(mass):
    '''
    Mass normlization function to determine symbol size
    This should be called both in sp_plot and sp_legend for the consistent result
    '''
    return np.sqrt(mass/300.)

def plt_proj_density(s, num, fig, savfig=True):
    """
    Create density projection
    """

    # create axes
    gs = GridSpec(1,3,figure=fig)
    ax1 = fig.add_subplot(gs[0:2])
    ax2 = fig.add_subplot(gs[2])
    cax1 = make_axes_locatable(ax1).append_axes('right', size='2.5%', pad=0.05)
    cax2 = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05)

    # load vtk and hst files
    ds = s.load_vtk(num=num)
    dat = ds.get_field(field=['density'], as_xarray=True)
    sp = read_starpar_vtk(s.files['starpar'][num])
    time = ds.domain['time']*s.u.Myr
    axis_idx = dict(x=0, y=1, z=2)

    # prepare variables to be plotted
    dat['surf_xy'] = (dat['density']*ds.domain['dx'][axis_idx['z']]).sum(dim='z')*s.u.Msun/s.u.pc**2
    dat['surf_xz'] = (dat['density']*ds.domain['dx'][axis_idx['z']]).sum(dim='y')*s.u.Msun/s.u.pc**2

    # plot

    # gas
    dat['surf_xy'].plot.imshow(ax=ax1, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e0, vmax=1e4, cbar_ax=cax1)
    dat['surf_xz'].plot.imshow(ax=ax2, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e-2, vmax=1e5, cbar_ax=cax2)

    # starpar
    young_sp = sp[sp['age']*s.u.Myr < 40.]
    young_cluster = young_sp[young_sp['mass'] != 0]
    mass = young_cluster['mass']*s.u.Msun
    age = young_cluster['age']*s.u.Myr
    cl = ax1.scatter(young_cluster['x1'], young_cluster['x2'], marker='o',
                     s=mass_norm(mass), c=age, edgecolor='black', linewidth=1,
                     vmax=40, vmin=0, cmap=plt.cm.cool_r, zorder=2, alpha=0.5)
    ax2.scatter(young_cluster['x1'], young_cluster['x3'], marker='o',
                s=mass_norm(mass), c=age, edgecolor='black', linewidth=1,
                vmax=40, vmin=0, cmap=plt.cm.cool_r, zorder=2, alpha=0.5)
    ss=[]
    label=[]
    ext = ax1.images[0].get_extent()
    for mass in [1e4,1e5,1e6]:
        ss.append(ax1.scatter(ext[1]*2,ext[3]*2,s=mass_norm(mass),color='k',alpha=.5))
        label.append(r'$10^%d M_\odot$' % np.log10(mass))
    ax1.set_xlim(ext[0], ext[1])
    ax1.set_ylim(ext[3], ext[2])
    ax1.legend(ss,label,scatterpoints=1,loc=2,ncol=3,bbox_to_anchor=(-0.1, 1.065), frameon=False)

    cax = fig.add_axes([0.15,0.93,0.25,0.015])
    cbar = plt.colorbar(cl, ticks=[0,20,40], cax=cax, orientation='horizontal')
    cbar.ax.set_title(r'$age\,[\rm Myr]$')

    for ax in [ax1,ax2]:
        ax.set_aspect('equal')

    # figure annotations
    fig.suptitle('{0:s}, time: {1:.1f} Myr'.format(s.basename, time), fontsize=30, x=.55, y=.93)

    if savfig:
        savdir = osp.join('./figures-all')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        fig.savefig(osp.join(savdir, 'proj-density.{0:s}.{1:04d}.png'
            .format(s.basename, ds.num)),bbox_inches='tight')

def plt_all(s, num, fig, with_starpar=False, savfig=True):
    """
    Create large plot including density slice, density projection, temperature
    slice, phase diagram, star formation rate, and mass fractions.
    """

    # create axes
    gs = GridSpec(3,5,figure=fig)
    ax1 = fig.add_subplot(gs[0,0])               # density slice
    ax2 = fig.add_subplot(gs[1:,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[0,1])               # density projection
    ax4 = fig.add_subplot(gs[1:,1], sharex=ax3)
    ax5 = fig.add_subplot(gs[0,2])               # temperature slice
    ax6 = fig.add_subplot(gs[1:,2], sharex=ax5)
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

    # load vtk and hst files
    ds = s.load_vtk(num=num)
    dat = ds.get_field(field=['density','pressure'], as_xarray=True)
    hst = s.read_hst(force_override=True)
    time = ds.domain['time']*s.u.Myr
    axis_idx = dict(x=0, y=1, z=2)
    
    # prepare variables to be plotted
    dat['pok'] = dat['pressure']*s.u.pok
    # T_1 = (p/k) / (rho m_p) is the temperature assuming mu=1
    dat['T1'] = dat['pok']/(dat['density']*s.u.muH)
    dat['temperature'] = xr.DataArray(coolftn().get_temp(dat['T1'].values),
            coords=dat['T1'].coords, dims=dat['T1'].dims)
    dat['surf_xy'] = ((dat['density']*ds.domain['dx'][axis_idx['z']])
            .sum(dim='z')*s.u.Msun/s.u.pc**2)
    dat['surf_xz'] = ((dat['density']*ds.domain['dx'][axis_idx['z']])
            .sum(dim='y')*s.u.Msun/s.u.pc**2)

    # plot

    # gas
    (dat['density'].interp(z=0)).plot.imshow(ax=ax1, norm=mpl.colors.LogNorm(),
            cmap='viridis', vmin=1e0, vmax=1e4, cbar_ax=cax1, add_labels=False,
            cbar_kwargs={'label':r"$n_{\rm H}\,[\rm cm^{-3}]$"})
    (dat['density'].interp(y=0)).plot.imshow(ax=ax2, norm=mpl.colors.LogNorm(),
            cmap='viridis', vmin=1e-2, vmax=1e4, cbar_ax=cax2, add_labels=False,
            cbar_kwargs={'label':r"$n_{\rm H}\,[\rm cm^{-3}]$"})
    dat['surf_xy'].plot.imshow(ax=ax3, norm=mpl.colors.LogNorm(), cmap='pink_r',
            vmin=1e0, vmax=1e4, cbar_ax=cax3, add_labels=False,
            cbar_kwargs={'label':r"$\Sigma_{\rm gas}\,[M_\odot\,\rm pc^{-2}]$"})
    dat['surf_xz'].plot.imshow(ax=ax4, norm=mpl.colors.LogNorm(), cmap='pink_r',
            vmin=1e-2, vmax=1e5, cbar_ax=cax4, add_labels=False,
            cbar_kwargs={'label':r"$\Sigma_{\rm gas}\,[M_\odot\,\rm pc^{-2}]$"})
    (dat['temperature'].interp(z=0)).plot.imshow(ax=ax5, norm=mpl.colors.LogNorm(),
            cmap='coolwarm', vmin=1e1, vmax=1e8, cbar_ax=cax5, add_labels=False,
            cbar_kwargs={'label':r"$T\,[\rm K]$"})
    (dat['temperature'].interp(y=0)).plot.imshow(ax=ax6, norm=mpl.colors.LogNorm(),
            cmap='coolwarm', vmin=1e1, vmax=1e8, cbar_ax=cax6, add_labels=False,
            cbar_kwargs={'label':r"$T\,[\rm K]$"})

    if with_starpar:
        # starpar
        try:
            sp = s.load_starpar_vtk(num=num)
            young_sp = sp[sp['age']*s.u.Myr < 40.]
            young_cluster = young_sp[young_sp['mass'] != 0]
            mass = young_cluster['mass']*s.u.Msun
            age = young_cluster['age']*s.u.Myr
            cl = ax3.scatter(young_cluster['x1'], young_cluster['x2'], marker='o',
                             s=mass_norm(mass), c=age, edgecolor='black', linewidth=1,
                             vmax=40, vmin=0, cmap=plt.cm.cool_r, zorder=2)
            ss=[]
            label=[]
            ext = ax3.images[0].get_extent()
            for mass in [1e4,1e5,1e6]:
                ss.append(ax3.scatter(ext[1]*2,ext[3]*2,s=mass_norm(mass),color='k',alpha=.5))
                label.append(r'$10^%d M_\odot$' % np.log10(mass))
            ax3.set_xlim(ext[0], ext[1])
            ax3.set_ylim(ext[3], ext[2])
            ax3.legend(ss,label,scatterpoints=1,loc=2,ncol=3,bbox_to_anchor=(0.0, 1.2), frameon=False)

            cax = fig.add_axes([0.15,0.93,0.25,0.015])
            cbar = plt.colorbar(cl, ticks=[0,20,40], cax=cax, orientation='horizontal')
            cbar.ax.set_title(r'$age\,[\rm Myr]$')
        except:
            # TODO this should catch proper exception
            pass

    # phase diagram
    histnP,xedgnP,yedgnP = np.histogram2d(
            np.log10(np.array(dat['density']).flatten()),
            np.log10(np.array(dat['pok']).flatten()), bins=200,
            range=[[-3,5],[2,8]],density=True,
            weights=np.array(dat['density']).flatten())
    histnT,xedgnT,yedgnT = np.histogram2d(
            np.log10(np.array(dat['density']).flatten()),
            np.log10(np.array(dat['temperature']).flatten()), bins=200,
            range=[[-3,5],[1,7]],density=True,
            weights=np.array(dat['density']).flatten())
    ax7.imshow(histnP.T, origin='lower', norm=mpl.colors.LogNorm(),
            extent=[xedgnP[0], xedgnP[-1], yedgnP[0], yedgnP[-1]], cmap='Greys')
    ax8.imshow(histnT.T, origin='lower', norm=mpl.colors.LogNorm(),
            extent=[xedgnT[0], xedgnT[-1], yedgnT[0], yedgnT[-1]], cmap='Greys')
    ax7.set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
    ax7.set_ylabel(r'$P/k_{\rm B}\,[{\rm K\,cm^{-3}}]$')
    ax8.set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
    ax8.set_ylabel(r'$T\,[{\rm K}]$')

    # history

    if "M001" in s.basename:
        Mdot = 0.01
        sfrlim = [1e-3,1e1]
        masslim = [1e4,1e6]
    elif "M0.1" in s.basename:
        Mdot = 0.1
        sfrlim = [1e-2,1e0]
        masslim = [1e5,1e7]
    elif "M1" in s.basename:
        Mdot = 1
        sfrlim = [1e-1,1e1]
        masslim = [1e6,1e8]
    elif "M10" in s.basename:
        Mdot = 10
        sfrlim = [1e0,1e2]
        masslim = [1e6,1e8]
    elif "V10" in s.basename:
        Mdot = 0.55 + 0.45*np.cos(2*np.pi*hst['time']/10)
        sfrlim = [5e-2,5e0]
        masslim = [5e5,5e7]
    elif "V50" in s.basename:
        Mdot = 0.55 + 0.45*np.cos(2*np.pi*hst['time']/50)
        sfrlim = [5e-2,5e0]
        masslim = [5e5,5e7]
    elif "V100" in s.basename:
        Mdot = 0.55 + 0.45*np.cos(2*np.pi*hst['time']/100)
        sfrlim = [5e-2,5e0]
        masslim = [5e5,5e7]
    elif "V200" in s.basename:
        Mdot = 0.55 + 0.45*np.cos(2*np.pi*hst['time']/200)
        sfrlim = [5e-2,5e0]
        masslim = [5e5,5e7]
    else:
        raise Exception("set appropriate yranges for the model {}".format(s.basename))

    ax9.semilogy(hst['time'], hst['sfr1'], 'm-', label='sfr1')
    ax9.semilogy(hst['time'], hst['sfr10'], 'r-', label='sfr10')
    ax9.semilogy(hst['time'], hst['sfr40'], 'g-', label='sfr40')
    ax9.semilogy(hst['time'], Mdot*np.ones(len(hst['time'])), 'k--', label='inflow')
    ax9.set_xlabel("time"+r"$\,[{\rm Myr}]$")
    ax9.set_ylabel("SFR"+r"$\,[M_\odot\,{\rm yr}^{-1}]$")
    ax9.set_ylim(sfrlim)
    ax9.plot([time,time],sfrlim,'y-',lw=5)
    ax9.legend()
    ax10.semilogy(hst['time'], hst['Mc'], 'b-', label=r"$M_c$")
    ax10.semilogy(hst['time'], hst['Mu'], 'g-', label=r"$M_u$")
    ax10.semilogy(hst['time'], hst['Mw'], 'r-', label=r"$M_w$")
    ax10.semilogy(hst['time'], hst['mass'], 'k-', label=r"$M_{\rm tot}$")
    ax10.semilogy(hst['time'], hst['msp'], 'k--', label=r"$M_{\rm sp}$")
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
    ax1.semilogy(hst['time'], hst['sfr1'], 'r-', label='sfr1')
    ax1.semilogy(hst['time'], hst['sfr5'], 'g-', label='sfr5')
    ax1.semilogy(hst['time'], hst['sfr10'], 'b-', label='sfr10')
    ax1.semilogy(hst['time'], hst['sfr40'], 'm-', label='sfr40')
    ax1.set_ylabel("SFR ["+r"$M_\odot\,{\rm yr}^{-1}$"+"]")
    ax1.set_ylim(sfrlim)
    ax1.legend()
    ax2.semilogy(hst['time'], hst['Mc'], 'b-', label=r"$M_c$")
    ax2.semilogy(hst['time'], hst['Mu'], 'g-', label=r"$M_u$")
    ax2.semilogy(hst['time'], hst['Mw'], 'r-', label=r"$M_w$")
    ax2.semilogy(hst['time'], hst['mass'], 'k-', label=r"$M_{\rm tot}$")
    ax2.semilogy(hst['time'], hst['msp'], 'k--', label=r"$M_{\rm sp}$")
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

def mass_flux(s, fig):
    """
    Check mass conservation of TIGRESS-GC simulation by measuring the net
    outflow rate and the total mass of the simulation box.
    Note that the nozzle inflow is already taken into account in the
    hst['F2_lower'] and hst['F2_upper'].

    ISSUE: We cannot check mass conservation accurately unless we use very 
    short time step between the history dump, because we have to time-integrate
    surface flux numerically - SMOON
    """
    # load history file
    hst = s.read_hst(force_override=True)
    Lx=s.par['domain1']['x1max']-s.par['domain1']['x1min']
    Ly=s.par['domain1']['x2max']-s.par['domain1']['x2min']
    Lz=s.par['domain1']['x3max']-s.par['domain1']['x3min']

    time = hst['time']
    sfr = hst['sfr10']
    Mtot = (hst['mass']+hst['msp'])
    inflow = np.ones(len(time))
    xflux = (-hst['F1_lower']+hst['F1_upper'])
    yflux = (-hst['F2_lower']+hst['F2_upper']) + inflow
    zflux = (-hst['F3_lower']+hst['F3_upper'])

    ax = fig.add_subplot(111)
    ax.semilogy(time, sfr, 'k-', label='SFR')
    ax.semilogy(time, xflux, 'b-', label='x1flux')
    ax.semilogy(time, yflux, 'r-', label='x2flux')
    ax.semilogy(time, zflux, 'g-', label='x3flux')
    ax.semilogy(time, inflow, 'k--', label='inflow rate')
    ax.semilogy(time, sfr+xflux+yflux+zflux, 'k:', label='SFR+fluxes')

def mass_conservation(s, fig):
    # load history file
    hst = s.read_hst(force_override=True)

    time = hst['time']
    sfr = hst['sfr10']
    Mtot = (hst['mass']+hst['msp'])
    Msp_left = hst['msp_left']
    xflux = (-hst['F1_lower']+hst['F1_upper'])
    yflux = (-hst['F2_lower']+hst['F2_upper'])
    zflux = (-hst['F3_lower']+hst['F3_upper'])
    outflow = xflux+yflux+zflux
    ax.semilogy(time, Mtot, 'k-')
    ax.semilogy(time, Mtot[0]-outflow*time-Msp_left, 'k--')
