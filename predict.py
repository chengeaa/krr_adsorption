###############
## paramters ##
###############

use_zhat = True # toggle for using KRR z model vs rule-based z model for test particle placement
use_test_particle = True #whether to use test particle approach vs mean kernel approach

test = True # set to true if testing accuracy of predictions is desired (ie, if an energies file exists for the target structure)
adslen = 5  # length of adsorbate

datadirs = ["data/adsorb_bombed_set1/"] # make sure that these structures *do* have adsorbates placed, even if there are no simulation energies!
data = preprocessE(datadirs, adslen = adslen, test = test) #again; start with **adsorbates** in the system; will produce test particles

zpath = "models/zhat.pkl" # pickle path + name for zmodel to save and/or read from
Epath = "models/Ehat.pkl" # pickle path + name for Emodel to save and/or read from

if use_test_particle:
    # input data has no guarantee that the initial z positions for the adsorbates are good, so give them a good start guess with rulez:
    for geom in Edata['processed']: 
        _x, _y, _z = geom[-1].position
        del geom[[atom.index for atom in geom if atom.index == len(geom)]]
        geom += Atom("He", position = [_x, _y, rulez(geom, _x, _y)])
    if use_zhat:
        with open(zpath, 'rb') as f:
            zhat = pickle.load(f)
        predicted_z_values = zhat.predict(Edata['processed'])
    else:
        predicted_z_values = [g[-1].position[2] for g in Edata['processed']
        # the alternative to not using zhat is to use the rule based placement - which we already did, so just grab those z values
        
    for i, g in enumerate(Edata['processed']):
        g[-1].position[2] = predicted_z_values[i] #again, assume last atom in each geom is the He test particle

    with open(Epath, 'rb') as f:
        Ehat = pickle.load(f)

    # now, use the placed test particles to perform the E estimation
    predicted_E_values = Ehat.predict(Edata['processed'])
    if test:
        y_test = Edata['E_ads']
        print("MAE(E)(test) = {}".format(mean_absolute_error(y_test, predicted_E_values)))
        print("MPAE(E)(test) = {}".format(mean_absolute_percentage_error(y_test, predicted_E_values)))
        plt.scatter(y_test, predicted_E_values)
        plt.plot(y_test, y_test)
        plt.savefig("prediction_test_results.png")
        plt.close()

else:
    # TODO: implement a model that doesn't use test particles

