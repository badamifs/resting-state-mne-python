# resting-state-mne-python

Resting State EEG workflow using mne-python https://www.martinos.org/mne/stable/index.html

# Prerequisites/Dependencies
Anaconda distribution https://www.anaconda.com/download/#macos

`curl -O https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml`

`conda env create -f environment.yml`

`source activate mne`

`conda install scipy matplotlib scikit-learn mayavi jupyter spyder`

`pip install PySurfer mne`

# Common errors 

The mayavi package sometimes does not install correctly via conda, try using pip for that package

# Verify installation 

From inside python call:

`>>> import mne`

# Jupyter notebook coming soon still working on source loc 
