"""
Created on Wed Feb 19 12:01:22 2025

@author: Santiago Isaac Flores Alonso
"""

#%% Import necessary libraries
import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy import stats



#%% Define paths for input and output data
meg_path = '/home/isaant/scratch/Cam-CAN/meg_outputs'  # Path where MEG outputs are stored
compacted_data_path = os.path.abspath(os.path.join(meg_path, 'compacted_data'))  # Path for compacted data
out_dir = os.path.abspath(os.path.join(meg_path, 'compacted_data'))  # Output directory for the processed data

LOO_neurochem_files = os.path.join(compacted_data_path, 'schaefer200_17networks_neurochems~*')
LOO_neurochem_files = np.sort(glob.glob(LOO_neurochem_files))

#%% Load subject IDs and prepare indices for lower triangular matrix extraction
subjects = np.sort(os.listdir(meg_path))[3:]  # Sort subjects and ignore the first two (probably system files)
lower_tri_indices = np.tril_indices(200, k=-1)  # Get lower triangular indices (excluding diagonal)

#%% Load precomputed clean scores and neurochemical correlation data
clean_scores = pd.read_csv(os.path.join(compacted_data_path, "clean_scores.csv"))  # Load clean behavioral scores

with open(os.path.join(compacted_data_path, "schaefer200_17networks_neurochems_correlation_df.pkl"), "rb") as f:
    neurochems_corr_df = pickle.load(f)  # Load neurochemical correlations dataframe

with open(os.path.join(compacted_data_path, "schaefer200_17networks_neurochems_partial_correlation_df.pkl"), "rb") as f:
    neurochems_partial_corr_df = pickle.load(f)  # Load neurochemical correlations dataframe


# Extract neurochemical correlation values for lower triangular indices
neurochems_corr = neurochems_corr_df['neurochems_corr'].to_numpy()[lower_tri_indices]
neurochems_partial_corr = neurochems_partial_corr_df['neurochems_partial_corr'].to_numpy()[lower_tri_indices]
neurochems_labels = neurochems_corr_df['labels']  # Neurochemical labels (unused in the code)
subjects_with_empty_dir=[]
#%% Iterate through each subject and build a dataframe with connectivity data
for i, subject in enumerate(subjects):
    print(i)
    if not os.listdir(os.path.join(meg_path, subject)):
        subjects_with_empty_dir.append(subject)
        continue
    # Initialize a DataFrame for storing connectivity and subject data
    sub_dataframe = pd.DataFrame(columns=[
        'ID', 'age', 'sex', 'handedness','age_full_time_edu_comp', 'degree', 'years_of_edu',
        'acer', 'cattell', 'interval', 'edge-label', 'delta', 'theta', 'alpha', 'beta',
        'g_low', 'g_high', 'from_ROI', 'to_ROI', 'hemi', 'neurochem_corr', 'euclidean_dist'
    ])

    # Load subject-specific functional connectivity correlations
    with open(os.path.join(meg_path, subject, "correlations_df.pkl"), "rb") as f:
        corr_df = pickle.load(f)

    # Extract subject scores
    sub_scores = clean_scores.loc[clean_scores['ID'] == subject]

    # Extract connectivity matrix, region labels, and centroid coordinates
    corr_matrix = corr_df['corr_matrix'][0]
    labels = corr_df['labels'][0][:200]  # Only use the first 200 labels
    centroids = corr_df['centroids'][0][:200]  # Only use the first 200 centroids

    # Convert region labels into formatted strings (removing prefixes)
    labels_names = np.array([label.name[14:-3] for label in labels])
    hemi = np.array([label.name[-2:] for label in labels])
    # Create a matrix of edge labels (each cell represents a region-to-region connection)
    labels_matrix = np.array([[f"{x}-{y}" for y in labels_names] for x in labels_names])[lower_tri_indices]
    hemi_matrix = np.array([[f"{x}-{y}" for y in hemi] for x in hemi])[lower_tri_indices]
    # Compute pairwise Euclidean distances between region centroids
    distance_matrix = cdist(centroids, centroids, metric='euclidean')[lower_tri_indices]

    # Populate the subject-specific dataframe
    sub_dataframe['ID'] = np.repeat(sub_scores['ID'].values, len(neurochems_corr))
    sub_dataframe['age'] = np.repeat(sub_scores['age'].values, len(neurochems_corr))
    sub_dataframe['sex'] = np.repeat(sub_scores['sex'].values, len(neurochems_corr))
    sub_dataframe['age_full_time_edu_comp'] = np.repeat(sub_scores['age_full_time_edu_comp'].values, len(neurochems_corr))
    sub_dataframe['degree'] = np.repeat(sub_scores['degree'].values, len(neurochems_corr))
    sub_dataframe['years_of_edu'] = np.repeat(sub_scores['years_of_edu'].values, len(neurochems_corr))
    sub_dataframe['handedness'] = np.repeat(sub_scores['handedness'].values, len(neurochems_corr))
    sub_dataframe['acer'] = np.repeat(sub_scores['acer'].values, len(neurochems_corr))
    sub_dataframe['cattell'] = np.repeat(sub_scores['cattell'].values, len(neurochems_corr))
    sub_dataframe['interval'] = np.repeat(sub_scores['interval'].values, len(neurochems_corr))

    # Assign connectivity edges and their properties
    sub_dataframe['edge-label'] = labels_matrix
    sub_dataframe['delta'] = stats.zscore(corr_matrix['delta'][:200, :200][lower_tri_indices])
    sub_dataframe['theta'] = stats.zscore(corr_matrix['theta'][:200, :200][lower_tri_indices])
    sub_dataframe['alpha'] = stats.zscore(corr_matrix['alpha'][:200, :200][lower_tri_indices])
    sub_dataframe['beta'] = stats.zscore(corr_matrix['beta'][:200, :200][lower_tri_indices])
    sub_dataframe['g_low'] = stats.zscore(corr_matrix['g_low'][:200, :200][lower_tri_indices])
    sub_dataframe['g_high'] = stats.zscore(corr_matrix['g_high'][:200, :200][lower_tri_indices])

    # Extract the "from" and "to" regions from edge labels
    sub_dataframe['from_ROI'] = [entry.split('-')[0] for entry in labels_matrix]
    sub_dataframe['to_ROI'] = [entry.split('-')[1] for entry in labels_matrix]
    sub_dataframe['hemi'] = hemi_matrix
    # Assign neurochemical correlation and Euclidean distance values
    sub_dataframe['neurochem_corr'] = neurochems_corr
    sub_dataframe['neurochem_partial_corr'] = neurochems_partial_corr
    for file in LOO_neurochem_files:
        col_name=os.path.basename(file).split('networks_')[1].split('_correlation_df')[0]
        with open(os.path.join(compacted_data_path, file), "rb") as f:
            neurochems_corr_LOO = pickle.load(f)  # Load neurochemical correlations dataframe
        neurochems_corr = neurochems_corr_LOO['neurochems_corr'].to_numpy()[lower_tri_indices]
        sub_dataframe[col_name] = neurochems_corr
    sub_dataframe['euclidean_dist'] = distance_matrix

    #save subjects csv into its own folder
    sub_dataframe.to_csv(os.path.join(out_dir,'schaefer200_17networks_neurochem-similarity_AEC-fc_all-subs_LOO', f"schaefer200_17networks_neurochem-similarity_AEC-fc_LOO_{subject}.csv"), index=False)

    #%% Append subject data to the full dataset
    if i == 0:
        full_dataframe = sub_dataframe.copy(deep=True)  # Initialize full DataFrame with first subject
        continue

    full_dataframe = pd.concat([full_dataframe,sub_dataframe],axis=0, ignore_index=True)
    if i % 10 == 0:
        full_dataframe.to_csv(os.path.join(out_dir, "schaefer200_17networks_neurochem-similarity_AEC-fc_LOO_full.csv"), index=False)


full_dataframe.to_csv(os.path.join(out_dir, "schaefer200_17networks_neurochem-similarity_AEC-fc_LOO_full.csv"), index=False)
print(len(full_dataframe), np.unique(full_dataframe['ID']))
subjects_with_empty_dir = pd.DataFrame(np.array(subjects_with_empty_dir))
subjects_with_empty_dir.to_csv(os.path.join(out_dir, "subjects_with_empty_dir.csv"),index=False, header=False)