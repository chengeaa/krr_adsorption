# KRR model for adsorption prediction

training data name format: batchname/output{i}.gen 
- "batchname" directory contains all assessed adsorption points on a single surface (example: ``bare_xtl_nrich`` would be a directory containing the results of assessing adsorption on many different points on a single bare, crystalline surface)
- each output{i}.gen file, then, should have that shared slab structure, with a unique adsorbate position represented. 

``train.py`` is main file for training a model; produces GridSearchCV objecst that are stored as pickles E model and z model, each (can be used for inference).
``predict.py`` is for applying the model to a new, unseen dataset. the predictions will be produced along with the errors (MAE,MPAE) for this new set (if true energies are present).

