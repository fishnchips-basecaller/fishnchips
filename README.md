# FishNChips - a CNN-Transformer basecaller
# Gravlax - a CNN-LSTM-CTC basecaller

## Required data

* folder with fast5 files for creating uids.json
    * multifast5 can be converted to fast5 using `multi_to_single_fast5` (`pip install ont_fast5_api`)
* folder with taiyaki outputs. We had multiple which allowed us to simply use the files themselves to ensure training was done on separate data than validating and testing
* Adapt reference file (`temps/ref-uniq.fa`) to one containing relevant DNA

## Requirements
* python3.7
* tensorflow==2.1
* mappy
* editdistance


## Running FishNChips

The FishNChips pipeline is run using the `run_fish.py` script, and the configurations can be passed as CL arguments. The script trains the model if not already trained and performs evaluation once training stops.

`python run-fish.py config.json experimentname`

sample configuration files are located in the `configs` folder and describe the configuration of the model, what bacteria are used for training, validation, and testing, and other parameters.

Testing can be done using concatenation or assembly, based on stride:

```
encoder_max_length == stride => concatenation
encoder_max_length > stride => assembly
```

## Running Gravlax

Gravlax is trained using `train_gravlax.py` and evaluated using `test_gravlax.py`. 

Gravlax does not have configurations, the parameters are hardcoded in the scripts. The naming of the output file is created based off the parameters, which helps distinguish models.
