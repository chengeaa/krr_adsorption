def predictz(surf, x, y, zmodel, species):
    """
    surf: bare substrate
    x, y: position at which to place adsorbate
    zmodel: model object

    returns predicted z value
    """
    searchR = 2.2
    surf = surf.copy()
    add_adsorbate(surf, 'He', height = 0, position = (x, y))

    maxz = 0
    for atom in surf:
        if atom.symbol == "He": # don't use He position to determine max Z position
            continue
        _x, _y, _z = atom.position
        if ((x - _x)**2 + (y - _y)**2) ** 0.5 < searchR:
            if _z > maxz:
                maxz = _z + 2.5

    surf[-1].position[2] = maxz

    X = getSOAPs(pd.Series({0: surf}), species = species)[0].reshape(1, -1) #reshape because just one sample
    print(X.shape)
    if zmodel:
        predz = zmodel.predict(X)
#       print(maxz, predz)
    else:
#         TODO: implement me hehe
        predz = maxz + 2.5
    return predz