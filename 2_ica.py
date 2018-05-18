"""Apply a filter to EEG data in a directory, and save the output in another
directory.
Command line example:
user$ python b_ica_all.py -i '1_firfilt' -o '2_firfilt_ica'
"""
import argparse
from glob import glob
from os import path as op
import mne
from mne import set_log_level, io
from mne.preprocessing import ICA
from mne.utils import ProgressBar

exir
def ica_all():
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
    parser.add_argument('-m', '--method', type=str, default='extended-infomax',
                        help='ICA method to use.')
    parser.add_argument('-v', '--verbose', type=str, default='error')
    args = parser.parse_args()

    input_dir = op.abspath(args.input)
    output_dir = op.abspath(args.output)
    ica_method = args.method

    if not op.exists(input_dir):
        sys.exit("Input directory not found.")
    if not op.exists(output_dir):
        sys.exit("Output directory not found.")

    set_log_level(verbose=args.verbose)

    input_fnames = op.join(input_dir, '*.fif')
    input_fnames = glob(input_fnames)
    n_files = len(input_fnames)

    print("Preparing to ICA {n} files".format(n=n_files))
    # Initialize a progress bar.
    progress = ProgressBar(n_files, mesg='Performing ICA')
    progress.update_with_increment_value(0)
    for fname in input_fnames:
        # Open file.
        raw = mne.io.read_raw_fif(fname, preload=True)
        # Perform ICA.
        ica = ICA(method=ica_method).fit(raw)
        # Save file.
        save_fname = op.splitext(op.split(fname)[-1])[0]
        save_fname += '-ica'
        save_fname = op.join(output_dir, save_fname)
        ica.save(save_fname + '.fif')
        # Update progress bar.
        progress.update_with_increment_value(1)

    print("")  # Get onto new line once progressbar completes.

if __name__ == "__main__":
    ica_all()