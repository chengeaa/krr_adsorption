from utils import *
from models import * 
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

########################
## control parameters ##
########################

# set datadirs to be list where the structures, energy files, and convergence files are
datadirs = ["data/adsorb_bombed_set0/", "data/adsorb_bombed_set1/"]
zpath = "models/zhat.pkl" # pickle path + name for zmodel to save and/or read from
Epath = "models/Ehat.pkl" # pickle path + name for Emodel to save and/or read from

use_zhat = True # if T, use KRR for z; if F, use rule for KRR
use_test_particle = True # if T, use the test particle approach; if F, use mean kernel

load_zmodel = False # train a new one and save to zpath
load_Emodel = False # train a new one and save to Epath

test = True # set to true if in train/test mode; False if training production model desired

kfolds = 5
CV_jobs = -1  # -1 means use all processors

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

verbosity = 1 #CV verbosity

######################################################
## use control parameters to get derived parameters ##
######################################################

# define grid search arrays for both models to search over
alphas = np.logspace(alphamin, alphamax, nalphas) # regularization term
gammas = np.linspace(gammamin, gammamax, ngammas) # kernel smoothness
sigmas = np.linspace(sigmamin, sigmamax, nsigmas) # SOAP smoothness
rcuts = np.linspace(rcutmin, rcutmax, nrcuts) # SOAP cutoff

Edata = preprocessE(datadirs, adslen = adslen) #returns filtered df (valid data) with ztrue and processed columns

# prevent model from using the initial positions, which are not known in general
for geom in Edata['processed']: 
    _x, _y, _z = geom[-1].position
    del geom[[atom.index for atom in geom if atom.index == len(geom)]]
    geom += Atom("He", position = [_x, _y, rulez(geom, _x, _y)])


############################
## begin training z model ##
############################
if use_test_particle:
    if use_zhat:
        if load_zmodel:
            with open(zpath, 'rb') as f:
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

            # fit z model
            zkrr = zmodel(species = species)
            zhat = GridSearchCV(zkrr, [{"alpha": alphas, "gamma": gammas, "sigma":sigmas, "rcut": rcuts}], 
                    cv = kfolds, n_jobs = CV_jobs, verbose = verbosity) 
            zhat.fit(X_train, y_train)
            print("z model fit:")
            print(zhat.best_params_)

            with open(zpath, 'wb') as f:
                pickle.dump(zhat, f)


    elif use_rule:
        pass
        # do nothing; the z rule is used to seed the initial position aleady
        # geometries = preprocessE(datadirs, adslen = adslen) #returns filtered df (valid data) with ztrue and processed columns
        # input_x_vals= [g[-1].position[0] for g in geometries]
        # input_y_vals  =  [g[-1].position[1] for g in geometries]
        # for g in geometries: # remove He atom
            # del g[[atom.index for atom in g if atom.index == len(g)]]
        # predicted_z_values = [rulez(surf, x, y) for surf in geometries['processed']]

    train_predicted_z_values = zhat.predict(X_train)
    zMAE = mean_absolute_error(y_train, train_predicted_z_values)
    zMPAE = mean_absolute_percentage_error(y_train, train_predicted_z_values)
    print("MAE(z) = {}".format(zMAE))
    plt.scatter(y_train, train_predicted_z_values)
    plt.plot(y_train, y_train)
    plt.savefig("z_model_train_results.png")
    plt.close()
    if test:
        test_predicted_z_values = zhat.predict(X_test)
        print("MAE(z)(test) = {}".format(mean_absolute_error(y_test, test_predicted_z_values)))
        plt.scatter(y_test, test_predicted_z_values)
        plt.plot(y_test, y_test)
        plt.savefig("z_model_test_results.png")
        plt.close()

    # after generating a predicted adsorbate z for each geom, set the adsorbate height with those predictions
    predicted_z_values = zhat.predict(Edata['processed']) 
    for i, g in enumerate(Edata['processed']):
        g[-1].position[2] = predicted_z_values[i] #again, assume last atom in each geom is the He test particle

else: # just use raw data without test particles
    for g in geometries:
        del g[[atom.index for atom in g if atom.index == len(g)]]

###################
## train E model ##
###################

if test:
    X_train, X_test, y_train, y_test = train_test_split(
        Edata['processed'], 
        Edata['E_ads'], random_state = 429)
    print("# points total: %d; #train points: %d; #test points: %d" % 
          (Edata.shape[0], len(X_train), len(X_test)))
else:
    X_train, y_train = Edata['processed'], Edata['E_ads']


alphas = np.logspace(alphamin, alphamax, nalphas)
gammas = np.linspace(gammamin, gammamax, ngammas)
sigmas = np.linspace(sigmamin, sigmamax, nsigmas)
rcuts = np.linspace(rcutmin, rcutmax, nrcuts)
Ekrr = Emodel(species = species)
ehat = GridSearchCV(Ekrr, [{"alpha": alphas, "gamma": gammas, "sigma":sigmas, "rcut":rcuts}], cv = kfolds, n_jobs = CV_jobs, verbose = verbosity) 
ehat.fit(X_train, y_train)
print("E model fit:")
print(ehat.best_params_)
with open(Epath, 'wb') as f:
    pickle.dump(ehat, f)
train_predicted_E_values = ehat.predict(X_train)

EMAE = mean_absolute_error(y_train, train_predicted_E_values)
EMPAE = mean_absolute_percentage_error(y_train, train_predicted_E_values)
print("MAE(E)(train) = {}".format(EMAE))
print("MPAE(E)(train) = {}".format(EMPAE))
plt.scatter(y_train, train_predicted_E_values)
plt.plot(Edata['E_ads'], Edata['E_ads'])
plt.savefig("E_model_train_results.png")
plt.close()
if test:
    test_predicted_E_values = ehat.predict(X_test)
    print("MAE(E)(test) = {}".format(mean_absolute_error(y_test, test_predicted_E_values)))
    print("MPAE(E)(test) = {}".format(mean_absolute_percentage_error(y_test, test_predicted_E_values)))
    plt.scatter(y_test, test_predicted_E_values)
    plt.plot(y_test, y_test)
    plt.savefig("E_model_test_results.png")
    plt.close()
