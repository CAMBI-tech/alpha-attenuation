Code to reproduce experiments in "Target-Related Alpha Attenuation in a Brain-Computer Interface Rapid Serial Visual Presentation Calibration"

# Setup

## Dependencies
Depends on python3.7+. See https://github.com/CAMBI-tech/BciPy for other transitive dependencies.

Create a virtualenv environment for the project and install required packages with:
```shell
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

# Usage
To request access to the data used for these experiments, please contact us at https://www.cambi.tech/contact.
Data should be stored in `data` with the following structure:
```shell
data/bcipy_recordings
├── <SUBJECT_ID_0>                                                       # Subject ID
│   ├── <SESSION_0_1Hz>                                                  # Recording session with 1 Hz stimulus
│   │   ├── parameters.json                                              # Experimental configuration
│   │   ├── raw_data.csv                                                 # Raw EEG data 
│   │   └── triggers.txt                                                 # Stimulus timing data
│   └── <SESSION_0_4Hz>                                                  # Recording session with 4 Hz stimulus
│       └── ...
├── <SUBJECT_ID_1>
│   └── ...
...
└── <SUBJECT_ID_N>
```

To run this code with data from another source, replace the `load_data` function in [alpha_experiment.py](alpha/alpha_experiment.py) with a function that loads data from the desired source.
Your function must produce data with shape `(trials, channels, samples)`.

Given a folder of data for a single subject, and an integer individual alpha frequency (IAF), reproduce experiments on alpha-band classifiers with:
```shell
source venv/bin/activate
python alpha/alpha_experiment.py --input <PATH/TO/DATA> --output <PATH/TO/RESULTS> --freq <IAF>
```

Similarly, to reproduce baseline experiments using BciPy's current PCA/RDA/KDE P300 classifier, run:
```shell
source venv/bin/activate
python alpha/baseline.py --input <PATH/TO/DATA> --output <PATH/TO/RESULTS>
```