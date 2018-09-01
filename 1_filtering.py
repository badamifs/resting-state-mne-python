#!/usr/bin/env python
"""
Apply a filter to EEG data in a directory, and save the output in another
directory.
Command line example::wi "data/0_easy" -o "data/1_firfilt" --lowpass 40.
--highpass .1
"""
import argparse
from glob import glob
from os import path as op
import mne
from mne import set_log_level
from mne.utils import ProgressBar

#from enobio import read_raw_enobio


def filter_all():
    """Filter all of the EEG data in a directory and save.
    Parameters
    ----------
    l_freq : float
        Low-pass frequency (Hz).
    h_freq : float
        High-pass frequency (Hz).
    read_dir : str
        Directory from which to read the data.
    save_dir : str
        Directory in which to save the filtered data.
    """
    parser = argparse.ArgumentParser(prog='1_filter_all.py',
                                     description=__doc__)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Directory of files to be filtered.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Directory in which to save filtered files.")
    parser.add_argument('-lp', '--lowpass', type=float, required=True,
                        help="Low-pass frequency (Hz).")
    parser.add_argument('-hp', '--highpass', type=float, required=True,
                        help="High-pass frequency (Hz).")
    parser.add_argument('-m', '--montage', default='Enobio32',
                        help='Electrode montage.')
    parser.add_argument('-ow', '--overwrite', type=bool, default='False',
                        help='If True, overwrites file if file exists.')
    parser.add_argument('-v', '--verbose', default='error')
    args = parser.parse_args()

    input_dir = op.abspath(args.input)
    output_dir = op.abspath(args.output)
    l_freq, h_freq = args.highpass, args.lowpass
    #montage = args.montage
    overwrite = args.overwrite

    if not op.exists(input_dir):
        sys.exit("Input directory not found.")
    if not op.exists(output_dir):
        sys.exit("Output directory not found.")

    set_log_level(verbose=args.verbose)

    input_fnames = op.join(input_dir, '*.mff')
    input_fnames = glob(input_fnames)
    n_files = len(input_fnames)

    print("Preparing to filter {n} files".format(n=n_files))
    # Initialize a progress bar.
    progress = ProgressBar(n_files, mesg='Filtering')

    failed_files = []
    for fname in input_fnames:
        try:
            raw = mne.io.read_raw_egi(fname, preload=True)
        except UserWarning:
            failed_files.append(fname)
            progress.update_with_increment_value(1)
            continue

        # High- and low-pass filter separately.
        raw.filter(l_freq=l_freq, h_freq=None, phase='zero',
                   fir_window='hamming', l_trans_bandwidth='auto',
                   h_trans_bandwidth='auto', filter_length='auto')
        raw.filter(l_freq=None, h_freq=h_freq, phase='zero',
                   fir_window='hamming', l_trans_bandwidth='auto',
                   h_trans_bandwidth='auto', filter_length='auto')

        # Create a new name for the filtered file.
        new_fname = op.split(fname)[-1]  # Remove path.
        new_fname = op.splitext(new_fname)[0]  # Remove extension.
        new_fname = new_fname[15:]  # Remove timestamp.
        new_fname = new_fname.replace("_Protocol 1", "")  # Remove Protocol 1.

        new_fname += '-firfilt'  # Indicate that we filtered.

        # Check for duplicates.
        base_name_to_check = op.join(output_dir, new_fname)
        if op.isfile(base_name_to_check + '.fif'):
            i = 1
            while op.isfile(base_name_to_check + '_{}.fif'.format(i)):
                i += 1
            new_fname += "_{}".format(i)

        raw.info['filename'] = new_fname  # Add this to the raw info dictionary.

        # Save the filtered file with a new name.
        save_fname = op.join(output_dir, new_fname)
        raw.save(save_fname + '.fif', overwrite=overwrite)

        # Update progress bar.
        progress.update_with_increment_value(1)

    print("")  # Get onto new line once progressbar completes.
    print("Failed on these files: {}".format(failed_files))

if __name__ == "__main__":
    filter_all()