#!/bin/env python

"""
Created on Wed Dec 25 22:23:53 2024
Description: Script to generate subject-level parcellated brains and fsaverage morphing
Author: Santiago Flores <sfloresa@sfu.ca>
"""

#%% Import necessary libraries
import os
import sys
import mne
import numpy as np
import matplotlib
import nibabel
import picard
import sklearn
import pickle
import pandas as pd
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from postprocessing import *

#%% Check if a subject is passed
if len(sys.argv) <= 1:
    raise ValueError("A subject directory has not been provided. Usage:"
                     "\n\tpython pipeline_rest_beamformer.py <subject_name>")
else:
    subject = sys.argv[1]
    session = sys.argv[2]

# Set number of cores for parallel processing (must be done before running any linear algebra functions)
num_cpu = '16'
os.environ['OMP_NUM_THREADS'] = num_cpu

#%% Directories
data_dir = '/home/isaant/scratch/BRSHN'
fs_dir = os.path.join(data_dir,'MRI/freesurfer') # Path to FreeSurfer subjects
meg_dir = os.path.join(data_dir,'/derivatives') # Path to MEG dataa
log_dir = os.path.join(data_dir, 'meg_outputs') 
output_dir = os.path.join(data_dir, 'meg_outputs', subject)
if not os.path.isdir(output_dir):
	os.mkdir(output_dir)

#%% List of required files
required_MRIfiles = ["lh.sphere.reg", "lh.pial", "lh.white", "lh.inflated",
                     "rh.sphere.reg", "rh.pial", "rh.white", "rh.inflated"]

required_MRIannot = ['lh.Schaefer2018_200Parcels_17Networks_order.annot', 
                     'rh.Schaefer2018_200Parcels_17Networks_order.annot'] # Must be added manually to the "label" for each subject and fsaverage6 folder from https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage6/label"

required_MEGfiles = [ "stc_beamformer-lh.stc", "stc_beamformer-rh.stc" ] # Generated stc

# Check if all required files exist
missing_MRIfiles = [ file for file in required_MRIfiles if not os.path.exists(os.path.join(fs_dir, subject, 'surf', file))]
missing_MRIannot = [ file for file in required_MRIannot if not os.path.exists(os.path.join(fs_dir, subject, 'label', file))]
missing_MEGfiles = [ file for file in required_MEGfiles if not os.path.exists(os.path.join(meg_dir, subject, file))]

# Open a file to save the logs
with open(os.path.join(log_dir,"missing_files_log.txt"), "a") as log_file:  # "a" mode appends to the file

    # Skip subject if files are missing
    if missing_MRIfiles or missing_MRIannot or missing_MEGfiles:
        missing_info = []
        if missing_MRIfiles:
            missing_info.append(f"MRI files: {', '.join(missing_MRIfiles)}")
        if missing_MRIannot:
            missing_info.append(f"MRI annot: {', '.join(missing_MRIannot)}")
        if missing_MEGfiles:
            missing_info.append(f"MEG files: {', '.join(missing_MEGfiles)}")

        # Prepare the message
        message = f"Skipping subject {subject} due to missing files: {', '.join(missing_info)}"
        
        # Print to console and save to log file
        log_file.write(message + "\n")  # Add a newline for each entry
        raise ValueError(message)
    
#%% Main steps
# Load the SourceEstimate for the subject
stc_path = os.path.join(meg_dir, subject, 'stc_beamformer-lh.stc')
src_path = os.path.join(meg_dir, subject, 'src_beamformer-src.fif')
stc = mne.read_source_estimate(stc_path)


# Parcellate time-series pca
labels = mne.read_labels_from_annot(subject, parc='Schaefer2018_200Parcels_17Networks_order', subjects_dir=fs_dir)# Read labels from parcellation file
src = mne.read_source_spaces(src_path)# Read source space 
centroids = calculate_parcellation_centroids(labels,src)

parc_ts_pca = mne.extract_label_time_course(stc, labels, src, mode='pca_flip').astype('float64') # Extract timeseries for parcellation 
parc_ts_df = pd.DataFrame({'parc_ts_pca': [parc_ts_pca],
                            'labels': [labels],
                            'centroids': [centroids]})



with open(os.path.join(output_dir,"parc_ts_Shaefer200_17Networks.pkl"), "wb") as f:
    pickle.dump(parc_ts_df, f)

#AEC

sfreq = stc.sfreq
correlations = compute_band_correlations(parc_ts_pca, sfreq, bands) #Function contained in postprocessing modules
correlations_df = pd.DataFrame({'corr_matrix': [correlations],
                            'labels': [labels],
                            'centroids': [centroids]})

with open(os.path.join(output_dir,"correlations_df.pkl"), "wb") as f:
    pickle.dump(correlations_df, f)


## ===============================================================================
## Morphing and PSD computation
## ===============================================================================

# In order to be able to average brains, a morph to standard space must be applied.
# Compute the morph to fsaverage6
morph = mne.compute_source_morph(
    stc,
    subject_from=subject,
    subject_to='fsaverage', # If you dont have it, try just fsaverage
    subjects_dir=fs_dir
)


# Compute normalized PSD and band powers

# Define frequency bands
bands = {
    'delta': (1, 4),
    'theta': (4, 7),
    'alpha': (8, 12),
    'beta': (15, 29),
    'g_low': (30, 59),
    'g_high': (60, 90)
}

psd_normalized, band_powers = PSD_per_vertex_parallel(stc,bands) #Function contained in postprocessing module

# Initialize storage for morphed band powers
band_powers_morph = None

# Loop through each frequency band
for i, (band, power) in enumerate(band_powers.items()):
    # Morph the band power to fsaverage
    stc_band_morph = stc_per_band(morph, power, stc, subject) #Function contained in postprocessing module

    # Initialize band power matrix on the first iteration
    if band_powers_morph is None:
        lh_vertices, rh_vertices = stc_band_morph.vertices
        total_vertices = len(lh_vertices) + len(rh_vertices)
        band_powers_morph = np.zeros((total_vertices, len(band_powers)))

    # Store the morphed data for the current band
    band_powers_morph[:, i] = stc_band_morph.data.flatten()

np.save(os.path.join(output_dir, "bands_power.npy"), band_powers_morph)

# Create and save spectrally resolved stc
stc_psd_morph = stc_per_band(morph, psd_normalized, stc, subject) #Function contained in postprocessing module
stc_psd_morph.tmin = 0
stc_psd_morph.tstep = .25
stc_psd_morph.save(os.path.join(output_dir, 'psd_beamformer_fsaverage'),overwrite=True)

# Parcellate spectrally resolved stc
stc_psd = mne.SourceEstimate(data=psd_normalized, vertices=stc.vertices, tmin=0, tstep=0.25, subject=subject)
parc_psd = mne.extract_label_time_course(stc_psd,labels, src, mode='mean').astype('float64') # Extract timeseries for parcellation ('mean' option avoids cancellation from default 'mean_flip' since MNE source activity is not signed)
parc_psd_df = pd.DataFrame({'parc_psd': [parc_psd],
                            'labels': [labels],
                            'centroids': [centroids]})

with open(os.path.join(output_dir,"parc_psd_Shaefer200_17Networks.pkl"), "wb") as f:
    pickle.dump(parc_psd_df, f)

