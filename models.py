import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from utils import getSOAPs

class zmodel(KernelRidge, RegressorMixin):
    """
    Class for predictive model that uses structural input and x,y location to predict a z height
    predicted z height is from sum of:
        utils.regionalMaxHeight (calculated)
        z offset from model (trained)
    Required args:
        -species (list of strings): contains the species that the descriptor needs to consider (be as inclusive as necesesary)
    """
    def __init__(
        self, species, alpha = 1, kernel = 'linear', gamma = None, degree = 3, coef0=1, kernel_params=None,
        rcut = 5, sigma = 0.1, zstep = 0.5, zrange = 2
    ):
        self.rcut = rcut
        self.gamma = gamma
        self.sigma = sigma
        self.zstep = zstep
        self.zrange = zrange
        self.species = species
        
        super(zmodel, self).__init__(
            alpha = alpha, kernel = kernel, gamma = gamma, 
            degree = degree, coef0 = coef0, kernel_params = None)

    def fit(self, data, y):
        """
        X is a pandas ``Series`` of the geometries after He substitution and range of z displacements
        y is the corresponding ``ztrue`` values
        """
        _soap = SOAP(
            species=self.species,
            periodic=True,
            rcut=self.rcut,
            nmax=10,
            lmax=9,
            sigma = self.sigma
        )

        # print('fitting with params \nalpha: {} \nsigma: {} \nrcut: {} \ngamma: {}\n'.format(self.alpha, self.sigma, self.rcut, self.gamma))
        
        data = pd.concat([data, getSOAPs(
            data, rcut = self.rcut, sigma = self.sigma, species = self.species)], axis = 1)
        X_agg = pd.concat([pd.DataFrame(data['SOAP'].to_list(), index = data.index), data], axis = 1)
        X = pd.concat([pd.DataFrame(data['SOAP'].to_list(), index = data.index)], axis = 1)
        
        # after preprocessing, use the regular KRR fit method
        super().fit(X, y)

    def predict(self, data):
        data = pd.concat([data, getSOAPs(
            data, rcut = self.rcut, sigma = self.sigma, species = self.species)], axis = 1)
        X = pd.concat([pd.DataFrame(data['SOAP'].to_list(), index = data.index)], axis = 1)
        return super().predict(X)
        
class Emodel(KernelRidge, RegressorMixin):
    """
    If jitter is being used, make sure the adsorbate (He) is the LAST ATOM in each structure
    A KRR model for predicting energy based on surface with single adsorbate. 
    """
    def __init__(
        self, species, alpha = 1, kernel = 'linear', gamma = None, degree = 3, coef0=1, kernel_params=None,
        rcut = 5, sigma = 0.1, usejitter = False, jitter = 0.25, 
    ):
        self.rcut = rcut
        self.sigma = sigma
        self.species = species
        self.jitter = jitter
        self.usejitter = usejitter
        
        super(Emodel, self).__init__(
            alpha = alpha, kernel = kernel, gamma = gamma, 
            degree = degree, coef0 = coef0, kernel_params = None)

    def fit(self, data, y):
        """
        data is a pandas ``Series`` of the He-substituted geometries
        y is the corresponding ``E_ads`` values
        """
        if self.usejitter:
            np.random.seed(429)
            for struct in data:
                # assumes that the LAST atom is the adsorbate atom
                struct[-1].position[2] += np.random.normal(scale = jitterscale) 
        _soap = SOAP(
            species=self.species,
            periodic=True,
            rcut=self.rcut,
            nmax=10,
            lmax=9,
            sigma = self.sigma
        )
        # print("number of soap features: {}".format(_soap.get_number_of_features()))

        data = pd.concat([data, getSOAPs(
            data, rcut = self.rcut, sigma = self.sigma, species = self.species)], axis = 1)
        X_agg = pd.concat([pd.DataFrame(data['SOAP'].to_list(), index = data.index), data], axis = 1)
        X = pd.concat([pd.DataFrame(data['SOAP'].to_list(), index = data.index)], axis = 1)
        # after preprocessing, use the regular KRR fit method
        super(Emodel, self).fit(X, y)
    def predict(self, data):
        data = pd.concat([data, getSOAPs(
            data, rcut = self.rcut, sigma = self.sigma, species = self.species)], axis = 1)
        X = pd.concat([pd.DataFrame(data['SOAP'].to_list(), index = data.index)], axis = 1)
        return super(Emodel, self).predict(X)
        



