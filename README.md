## Blind image quality model using RNN for spatial pooling (RNN-BIQA)

We propose an image quality model attempting to mimic the attention mechanism of human visual system (HVS) by using a recurrent neural network (RNN) for spatial pooling of the features extracted from different spatial areas (patches) by a deep CNN-based feature extractor. This package contains the essential Matlab scripts for implementing the proposed image quality assessment method.

As a prerequisite, the following third-party image quality databases need to be installed:

LIVE Challenge image quality database from: http://live.ece.utexas.edu/research/ChallengeDB/

KoNIQ-10k image quality database from: http://database.mmsp-kn.de/koniq-10k-database.html

SPAQ image quality database from: https://github.com/h4nwei/SPAQ

For using the implementation, extract all the Matlab scripts (*.m) in the same folder.

For training and testing the model from scratch, you can use `masterScript.m`. It can be run from 
Matlab command line as:

```
>> masterScript(livec_path, koniq_path, spaq_path, cpugpu);
```

The following input is required:

`livec_path`: path to the LIVE Challenge dataset, including metadata files _allmos_release.mat_ and 
_allstddev_release.mat_. For example: _'c:\\livechallenge'_.

`koniq_path`: path to the KoNIQ-10k dataset, including metadata file 
_koniq10k_scores_and_distributions.csv_. For example: _'c:\\koniq10k'_.

`spaq_path`: path to the SPAQ dataset, including metadata _file mos_spaq.xlsx_. For example: 
_'c:\\spaq'_.

`cpugpu`: whether to use CPU or GPU for training and testing the models, either _'cpu'_ or _'gpu'_.

The script implements the following functionality:

1) Makes patches out of LIVE Challenge dataset and makes probabilistic quality scores (file 
_LiveC_prob.mat_), `using processLiveChallenge.m` script.
2) Makes downscaled version of the SPAQ dataset (SPAQ-768), using `resizeImages.m` script.
3) Trains CNN feature extractor, using `trainCNNmodel.m` script.
4) Extracts feature vector sequences from KoNIQ-10k and SPAQ images, using the trained
feature extractor and `computeCNNfeatures.m` script.
5) Trains and tests RNN model by using KoNIQ-10k features for training and SPAQ for testing,
and then vice versa. Uses `trainAndTestRNNmodel.m` script for this purpose. Displays the results
for SCC, PCC, and RMSE.
