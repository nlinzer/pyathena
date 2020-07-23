import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .read_hst import ReadHst
# from .read_zprof import ReadZprof
# from .plt_hst_zprof import PltHstZprof

class LoadSimTIGRESSGC(LoadSim, ReadHst): #, ReadZprof, PltHstZprof):
    """LoadSim class for analyzing TIGRESS-GC simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 verbose=False):
        """The constructor for LoadSimTIGRESSGC class

        Parameters
        ----------
        basedir : str
            Name of the directory where all data is stored
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str
            Load vtk using 'pyathena' or 'yt'. Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the string
            representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.
        """

        super(LoadSimTIGRESSGC,self).__init__(basedir, savdir=savdir,
                                               load_method=load_method, verbose=verbose)

        # Set unit
        self.u = Units(muH=1.4271)

class LoadSimTIGRESSGCAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()
            # M0.1
            models['M0.1_1pc'] = '/data/shmoon/TIGRESS-GC/M0.1_1pc'
            models['M0.1_2pc'] = '/data/shmoon/TIGRESS-GC/M0.1_2pc'
            models['M0.1_4pc'] = '/data/shmoon/TIGRESS-GC/M0.1_4pc'
            # M1
            models['M1_1pc'] = '/data/shmoon/TIGRESS-GC/M1_1pc'
            models['M1_2pc'] = '/data/shmoon/TIGRESS-GC/M1_2pc'
            models['M1_4pc'] = '/data/shmoon/TIGRESS-GC/M1_4pc'
            # M10
            models['M10_1pc'] = '/data/shmoon/TIGRESS-GC/M10_1pc'
            models['M10_2pc'] = '/data/shmoon/TIGRESS-GC/M10_2pc'
            models['M10_4pc'] = '/data/shmoon/TIGRESS-GC/M10_4pc'

        self.models = list(models.keys())
        self.basedirs = dict()
        
        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        
        self.model = model
        self.sim = LoadSimTIGRESSGC(self.basedirs[model], savdir=savdir,
                                    load_method=load_method, verbose=verbose)
        return self.sim
