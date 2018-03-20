import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import io 
from mne.epochs import EpochsArray
from mne.event import define_target_events
from mne.datasets import sample
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import ICA


# getting some data ready
data_path = ''
raw_fname = data_path + '.mff'

raw = mne.io.read_raw_egi(raw_fname, preload=True)
info = raw.info

raw.info['bads'] = ['E257']

#remove bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                       exclude='bads')

#Epoch data 
n_epochs=100
Fs = raw.info['sfreq']
rdata = raw.get_data()
events = np.array([[int(i*rdata.shape[1]/n_epochs), 0, 1] for i in range(n_epochs)])
epochs = mne.Epochs(raw, events, 1, 0, int(0.9*(rdata.shape[1]/n_epochs)/Fs), proj=True, picks=picks)
print(epochs)
epochs_data = epochs.get_data()
print(epochs_data.shape)
#io.savemat('epochs_data.mat', dict(epochs_data=epochs_data), oned_as='row')

#evoked data 
evoked = epochs.average()
print(evoked)
evoked.plot()
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5))

# tmin, tmax = 0, 20  # use the first 20s of data

# fmin, fmax = 2, 300  # look at frequencies between 2 and 300Hz
# n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2

print(raw)
print(raw.info)
print(raw.ch_names)

# start, stop = raw.time_as_index([100, 115]) # 100 s to 115 s data segment
# data, times = raw[:, start:stop]
# print(data.shape)
# print(times.shape)
# data, times = raw[2:20:3, start:stop]  # access underlying data

#raw.info['bads'] = ['Status']

#remove bad channels
#picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,

#print('channels selected')
#print(picks)

# raw.crop(tmin, tmax).load_data()

#Let's first check out all channel types
plot_psd = raw.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=False)
plot_psd.savefig('raw_psd.png')

plot_raw = raw.plot(block=True, lowpass=40)
plot_raw.savefig('raw_data.png')

#Notch filtering (pwerline)
EEG = raw.notch_filter(np.arange(56, 90, 64), picks=picks, filter_length='auto',
                 phase='zero')
notch_filtered_psd = EEG.plot_psd(area_mode='range', picks=picks, tmax=10.0, average=False)
notch_filtered_psd.savefig('notch_raw.png')

# low pass filtering below 50 Hz
EEG = raw.filter(None, 50., fir_design='firwin')
lowpass = EEG.plot_psd(area_mode='range', tmax=10.0, average=False)
lowpass.savefig('lowpass_eeg.png')

# high pass filtering 
EEG = raw.filter(1., None, fir_design='firwin')
highpass = raw.plot_psd(area_mode='range', tmax=10.0, average=False)
highpass.savefig('highpass_eeg.png')

EEG = raw.resample(256, npad="auto")  # set sampling frequency to 100Hz
resampled = raw.plot_psd(area_mode='range', tmax=10.0, picks=picks)
resampled.savefig('resampled_data.png')

# # plot butterfly
# butterfly = raw.plot(butterfly=True, group_by='position')

# ntrial, nchan, ntime = 100, 73, 10
# events = np.zeros((ntrial, 3))
# np.random.seed(0)
# events[:, 2] = np.random.randn(1, ntrial)
# events[:, 2] = np.round(1e7 * np.random.randn(1, ntrial))
# print(events)

# EpochsArray(raw, info, events=events, verbose=False)

#ICA parameters

n_components = 25  # if float, select n_components by explained variance of PCA
method = 'extended-infomax'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23

ica = ICA(n_components=n_components, method=method, random_state=random_state)
print(ica)

#we avoid fitting ICA on crazy environmental artifacts that would dominate the variance and decomposition

reject = dict(mag=5e-12, grad=4000e-13)
ica.fit(raw, picks=picks, decim=decim, reject=reject)
print(ica)

#ica.plot_components()  # can you spot some potential bad guys?

#ica.plot_properties(raw, picks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,], psd_args={'fmax': 35.})

#ica.save('best-saline-ica.fif')

raw_copy = raw.copy()
print(raw_copy)
ica.apply(raw_copy)
raw_copy.plot()  # check the result

# events = np.array([
#     [0, 0, 1],
#     [1, 0, 2],
#     [2, 0, 1],
#     [3, 0, 2],
#     [4, 0, 1],
#     [5, 0, 2],
#     [6, 0, 1],
#     [7, 0, 2],
#     [8, 0, 1],
#     [9, 0, 2],
# ])

# event_id = dict(smiling=1, frowning=2)

# epochs = mne.Epochs(raw, events, tmin, tmax, proj=True, picks=picks,
#                     preload=False)
# print(epochs)


# # events = mne.read_events(event_fname)

# # reference_id = 5  # presentation of a smiley face
# # target_id = 32  # button pre
# # sfreq = raw.info['sfreq']  # sampling rate
# # tmin = 0.1  # trials leading to very early responses will be rejected
# # tmax = 0.59  # ignore face stimuli followed by button press later than 590 ms
# # new_id = 42  # the new event id for a hit. If None, reference_id is used.
# # fill_na = 99  # the fill value for misses

# # events_, lag = define_target_events(events, reference_id, target_id,
# #                                     sfreq, tmin, tmax, new_id, fill_na)

# # print(events_)

# # event_id, tmin, tmax = 9, -0.2, 0.5

# # EEG = mne.Epochs(EEG,3)

# plot butterfly
#butterfly = raw.plot(butterfly=True, group_by='position')
