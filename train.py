from utils import *
from models import * 
import pickle

# set datadirs to be list where the structures, energy files, and convergence files are
datadirs = ["data/adsorb_bombed_set0/", "data/adsorb_bombed_set1/"]
zpath = "models/zhat.pkl" # pickle path + name for zmodel to save and/or read from
Epath = "models/Ehat.pkl" # pickle path + name for Emodel to save and/or read from

use_zhat = True # if T, use KRR for z; if F, use rule for KRR
use_test_particle = True # if T, use the test particle approach; if F, use mean kernel

load_zmodel = False # train a new one and save to zpath
load_Emodel = False # train a new one and save to Epath

test = False # set to true if in test mode

kfolds = 5
CV_jobs = 4 

adslen = 5 # number of atoms in adsorbate molecule
zrange = 1 # 
zstep = 0.5

species = ["Si", "N", "H", "C", "F", "He"] #He is the test particle species
nalphas = 2 #number of alpha values to search over in CV
alphamin, alphamax = -10, -1
nsigmas = 2 #number of sigma values to search over in CV
sigmamin, sigmamax = 0.1, 1
ngammas = 2 #number of gamma values to search over in CV
gammamin, gammamax = 0.1, 1
nrcuts = 2 #number of rcut values to search over in CV
rcutmin, rcutmax = 2, 10

Edata = preprocessE(datadirs, adslen = adslen) #returns filtered df (valid data) with ztrue and processed columns

if use_test_particle:
    if use_zhat:
        if load_zmodel:
            with open(zpath, 'wb') as f:
                zhat = pickle.load(f)
        else:
            data = preprocessz(datadirs, zrange = zrange, zstep = zstep, adslen = adslen) #returns filtered df (valid data) with ztrue and processed columns

            if test:
                X_train, X_test, y_train, y_test = train_test_split(
                    data['processed'], 
                    data['ztrue'], random_state = 429)
                print("# points total: %d; #train points: %d; #test points: %d" % 
                      (data.shape[0], len(X_train), len(X_test)))
            else:
                X_train, y_train = data['processed'], data['ztrue']

            # define grid search arrays for zmodel to search over
            alphas = np.logspace(alphamin, alphamax, nalphas) # regularization term
            gammas = np.linspace(gammamin, gammamax, ngammas) # kernel smoothness
            sigmas = np.linspace(sigmamin, sigmamax, nsigmas) # SOAP smoothness
            rcuts = np.linspace(rcutmin, rcutmax, nrcuts) # SOAP cutoff

            # fit z model
            zkrr = zmodel(species = species)
            zhat = GridSearchCV(zkrr, [{"alpha": alphas, "gamma": gammas, "sigma":sigmas, "rcut": rcuts}], 
                    cv = kfolds, n_jobs = CV_jobs) 
            zhat.fit(X_train, y_train)

            with open(zpath, 'rb') as f:
                pickle.dump(zhat, f)

            predicted_z_values = zmodel.predict(Edata['processed']) 

    elif use_rule:
        geometries = preprocessE(datadirs, adslen = adslen) #returns filtered df (valid data) with ztrue and processed columns
        for g in geometries: # remove He atom
            del g[[atom.index for atom in g if atom.index == len(g)]]
        predicted_z_values = [rulez(surf) for surf in geometries['processed']]

    # after generating a predicted adsorbate z for each geom, set the adsorbate height with those predictions
    for i, g in enumerate(Edata['processed']):
        g[-1].position[2] = predicted_z_values[i] #again, assume last atom in each geom is the He test particle


else:
    # just use raw data without test particles
    for g in geometries:
        del g[[atom.index for atom in g if atom.index == len(g)]]

print(predicted_z_values)

# train E model

# X_train, X_test, y_train, y_test = train_test_split(
    # data['processed'], 
    # data['E_ads'], random_state = 429)
# print("# points total: %d; #train points: %d; #test points: %d" % 
      # (data.shape[0], len(X_train), len(X_test)))


# alphas = np.logspace(-10, -1, 25)
# gammas = np.linspace(-1, 1, 25)
# sigmas = np.linspace(-1, 1, 25)
# Ekrr = Emodel()
# ehat = GridSearchCV(Ekrr, [{"alpha": alphas, "gamma": gammas, "sigmas"}], cv = kfolds, n_jobs = CV_jobs) 
# ehat.fit(X_train, y_train)
# with open(Epath, 'rb') as f:
    # pickle.dump(ehat, f)
        
        # pickle.dump(zmodel, f)
