import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .read_hst import ReadHst

class LoadSimFeedbackTest(LoadSim, ReadHst):
    """LoadSim class for analyzing LoadSimFeedbackTest simulations.
    """
    
    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 units=Units(kind='LV', muH=1.4271),
                 verbose=False):
        """The constructor for LoadSimFeedbackTest class

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

        super(LoadSimFeedbackTest,self).__init__(basedir, savdir=savdir,
                                                 load_method=load_method,
                                                 units=units,
                                                 verbose=verbose)


class LoadSimFeedbackTestAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()
            models['newcool.n200.M1E3.N128'] = '/perseus/scratch/gpfs/jk11/FEEDBACK-TEST/roe.newcool.n200.M1E3.N128'

        self.models = list(models.keys())
        self.basedirs = dict()
        
        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        
        self.model = model
        self.sim = LoadSimFeedbackTest(self.basedirs[model], savdir=savdir,
                                       load_method=load_method, verbose=verbose)
        return self.sim