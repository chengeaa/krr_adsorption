import os
import re
from dscribe.descriptors import SOAP
from sklearn import preprocessing
from ase import Atom, Atoms
from ase.io import gen
import numpy as np
import pandas as pd

def readStructs(datadir, shallow = True, name = "output"):
    """
        Currently designed for output from single layer directory trees.
        Reads in final adsorption geometries and energy data, returns dataframe with geometry and energy data

        Input:
            datadir: string that points to directory containing the following:
                - convergence: each line i has convergence status of run i
                - energies: each line i has total energy and ads energy from run i
                - output{indices}.gen: final geometries for each index
                
            slabEnergy: energy of slab
            adsorbateEnergy: energy of the adsorbate in the system

        Returns:
            output: pd Dataframe with:
                - index: indices for runs that worked
                - geometry: final geometry of run
                - total energy: raw energy from file
                - adsorption energy: energy as adjusted by adsorbate_energy
    """
    geometries = {}
    if shallow:
        pattern = r"{}(\d+).gen".format(name)
    else:
        pattern = r"{}(\d+-\d+).gen".format(name)
    files = os.listdir(datadir) 

    if "energies" in files and "convergence" in files:
        convergence = pd.read_csv(datadir + "convergence", header = None)
        energies = pd.read_csv(datadir + "energies", header = None)
        output =  pd.concat([energies, convergence], axis = 1)
        output.columns = ["E", "E_ads", "conv"]

        for i in files:
            key = re.search(pattern, i)
            if key:
                if shallow:
                    key = int(key.group(1))
                else:
                    key = key.group(1)
                geometries[key] =  gen.read_gen(datadir + i)
        output['geom'] = pd.Series(geometries)

        output = output[output['conv'] == "Geometry converged"]
        output = output.drop("conv", axis = 1)

    else:
        for i in files:
            key = re.search(pattern, i)
            if key:
                if shallow:
                    key = int(key.group(1))
                else:
                    key = key.group(1)
                geometries[key] =  gen.read_gen(datadir + i)
        output = pd.DataFrame(pd.Series(geometries))
        output.columns = ['geom']
    return output

def regionalMaxHeight(surf, x, y, R = 2.2):
    """
    Returns the max height of any atom in a region of radius R around x,y
    R is default 2.2 from some experiments I did
    """
    maxz = 0
    for atom in surf:
        if atom.symbol == "He": # don't use He position to determine max Z position
            continue
        _x, _y, _z = atom.position
        if ((x - _x)**2 + (y - _y)**2) ** 0.5 < R:
            if _z > maxz:
                maxz = _z 
    return maxz

def predictz(surf, x, y, zmodel, species):
    """
    surf: *bare* substrate
    x, y: position at which to place adsorbate
    zmodel: model object (which takes in a dataframe that's n*p, p = #SOAP features)

    returns a predicted z value, based on sum of regionalMaxHeight and zmodel outcome
    """
    searchR = 2.2
    surf = surf.copy()
    add_adsorbate(surf, 'He', height = 0, position = (x, y))

    surf[-1].position[2] = rulez(surf, x, y, species) # use rulez for initial guess for z model

    X = getSOAPs(pd.Series({0: surf}), species = species)[0].reshape(1, -1) #reshape because just one sample
    predz = regionalMaxHeight(surf, x, y, species) + zmodel.predict(X) 
    return predz

def rulez(surf, x, y):
    """
    surf: *bare* substrate
    x, y: position at which to place adsorbate

    returns predicted z value
    """
    surf = surf.copy()

    return regionalMaxHeight(surf, x, y) + 2.5

def convertAdsorbateToHe(struct, centerIndex, molIndices, height = None):
    """
    Preprocess final relaxed adsorption structures; replace adsorbate with He

    Args:
        - struct: total structure (Atoms object)
        - centerIndex: index of central atom (where He will be) (int)
        - molIndices: list of indices to delete from the slab
        - height(float) : height of He to be placed
    Returns:
        - output: Atoms object with He representing the location of the adsorbate
    """
    x, y, z = struct[centerIndex].position
    output = struct.copy()
    del output[[atom.index for atom in output if atom.index in molIndices]]
    if height:
        add_adsorbate(output, "He", height = height, position = (x, y))
    else:
        output.append(Atom("He", position=[x,y,z])) # adds to exact position of centeratom
    return output


def getSOAPs(geometries, species,
        rcut, sigma, nmax = 10, lmax = 9, 
             periodic = True, crossover = True, sparse = False):
    """
    Takes a Series of geometries with one He present,
        returns SOAP representation of the chemical environment of He for each item
    Assumes any given structure in ``geometries`` has the same collection of elements
        as all the other structures
    Assumes any given structure in ``geometries`` has the same number of atoms as all
        the other structures

    Input:
        geometries: Series of Atoms objects; each must contain exactly 1 He atom
        rcut, nmax, lmax, sigma, periodic, crossover, sparse: SOAP parameters
    Output:
        output: Series of SOAP matrices, each corresponding to the appropriate index
    """
#   refgeom = geometries.iloc[0] #use the first geometry as a reference geometry

    ## set up descriptor
#   species = np.unique([i.symbol for i in refgeom])
    desc = SOAP(species=species, rcut = rcut, nmax = nmax, lmax = lmax,
                sigma = sigma, periodic = periodic, crossover = crossover, sparse = sparse)
    ## apply descriptor
    soaps = {}
    for i, geom in geometries.iteritems():
        HeLoc = len(geom) - 1  # assume He atom is last one in Atoms list
        tempSOAP = preprocessing.normalize(
            desc.create(geom, positions = [HeLoc], n_jobs = 4)) # SOAP representation of temp
        soaps[i] = tempSOAP[0]
    return pd.Series(soaps,name = 'SOAP')


def preprocessE(datadirs, adslen = 5, test = False):
    """
    Preprocess data for E model with the following steps:
        Converts adsorbate molecule (last in the Atoms list) to He
        Filters out energies that are nonnegative (assumes adsorption should be at least slightly favorable, thus E < 0)
    Inputs: 
        - datadirs (list of strs): directories with .gen structures, ``energies``, and ``convergence`` files
        - adslen: length of adsorbate in the system (ie, number of atoms in adsorbate)
        - test: if this dataset should be used to evaluate an accuracy (ie, if simulation-calculated adsorption energies exist, read them in)
    Returns:
        - data (pd df): has columns 'E_ads' and 'processed'\ 
        with the adsorption energy and structure with He substituted, respectively.
    """
    processed = []
    for datadir in datadirs:
        tempdata = readStructs(datadir)
        tempdata['processed'] = pd.Series(
            {key: convertAdsorbateToHe(i, len(i) - adslen, np.arange(len(i) - adslen, len(i))) for key, i in tempdata['geom'].items()
        })
        processed += [tempdata]
    data = pd.concat(processed).reset_index(drop = True).fillna(0)
    if test:
        validData = data['E_ads'] < 0  # adsorption energies should be negative
        return data.loc[validData, ["E_ads", "processed"]]
    else:
        return data.loc[:, ["processed"]]

def preprocessz(datadirs, zrange = 2, zstep = 0.25, adslen = 5):
    """
    Preprocess data for zmodel with the following steps:
        Converts adsorbate molecule (last in the Atoms list) to He
        Filters out energies that are nonnegative (assumes adsorption should be at least slightly favorable, thus E < 0)
    Inputs: 
        - datadirs (list of strs): directories with .gen structures, ``energies``, and ``convergence`` files
    Returns:
        - data (pd df): has columns 'ztrue' and 'processed'\ 
        with the adsorption energy and structure with He substituted, respectively.
        Given n*p input, output will be (n*zrange/zstep)*p 
    """
    zdiffs = np.arange(-zrange, zrange, zstep)
    processed = []
    for datadir in datadirs:
        tempdata = readStructs(datadir)
        tempdata['processed'] = pd.Series(
            {key: convertAdsorbateToHe(i, len(i) - adslen, np.arange(len(i) - adslen, len(i))
                                                   ) for key, i in tempdata['geom'].items()
        })
        processed += [tempdata]
    data = pd.concat(processed).reset_index(drop = True).fillna(0)
    validData = data['E_ads'] < 0  # adsorption energies should be negative
    data = data.loc[validData, :]
    data['ztrue'] = [s[-1].position[2] for s in data['processed']]
    newdata = data.copy()
    # newdata = newdata.iloc[0:0]

    for idx, row in data.iterrows():
        ztrue = row['ztrue']
        for zdiff in zdiffs:
            newrow = row.copy()
            tempStruct = newrow['processed'].copy()
            tempStruct[-1].position[2] = ztrue + zdiff
            newrow['processed'] = tempStruct
            newdata = newdata.append(newrow) 
        
    newdata = newdata.reset_index(drop = True)

    return newdata.loc[:, ["ztrue", "processed"]]


