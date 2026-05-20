#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 22:46:26 2024

@author: Santiago Isaac Flores Alonso
"""
#%%
import pandas as pd
import numpy as np
import os
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import re
import seaborn as sns
import pickle as pkl

#%%
def plot_neurocorr_matrix(corr, labels, out_dir):
    """
    Plot connectivity matrices with optional reordering and boundary marking.

    Parameters:
        corr (numpy array): Connectivity matrix.
        labels (list): List of atlas labels.
    """
    Vis = ['VisCen', 'VisPer']
    change_of_net_pos = []
    current_net = re.search(r'H_(.*?)_', str(labels[0])).group(1)
    all_nets = [re.search(r'H_(.*?)_', str(labels[0])).group(1)]
    for i, label in enumerate(labels[1:]):
        match = re.search(r'H_(.*?)_', str(label)).group(1)
        if current_net != match:
            current_net = match
            change_of_net_pos.append(i + 1)
            all_nets.append(match)
            # if len(change_of_net_pos) > 3 and all_nets[1] == all_nets[-1]:
            #   break

    change_of_supnet_pos = []
    current_net = re.search(r'H_(.*?)_', str(labels[0])).group(1)[:-1]
    all_supnets = [re.search(r'H_(.*?)_', str(labels[0])).group(1)[:-1]]
    for i, label in enumerate(labels[1:]):
        match = re.search(r'H_(.*?)_', str(label)).group(1)[:-1]
        if match == Vis[0] or match == Vis[1]:
            match = 'Vis'
        if current_net != match:
            current_net = match
            change_of_supnet_pos.append(i + 1)
            all_supnets.append(match)

    
    

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    clim = np.percentile(corr, [0, 100])
    sns.heatmap(corr, cmap="GnBu", vmin=clim[0], vmax=clim[1], ax=ax, cbar=True)
    ax.set_title("Pairwise correlation: Neurochems system")

    ax.hlines(100, *ax.get_xlim(), colors="black", linewidth=2.0, linestyles="dashed")
    ax.vlines(100, *ax.get_ylim(), colors="black", linewidth=2.0, linestyles="dashed")

    start = 0
    end = 200
    for i in range(len(change_of_net_pos) - 1):
        ax.hlines(change_of_net_pos[i], start, change_of_net_pos[i + 1], colors="red", linewidth=2.0)
        ax.vlines(change_of_net_pos[i], start, change_of_net_pos[i + 1], colors="red", linewidth=2.0)
        start = change_of_net_pos[i]
    ax.hlines(change_of_net_pos[-1], start, end, colors="red", linewidth=2.0)
    ax.vlines(change_of_net_pos[-1], start, end, colors="red", linewidth=2.0)

    start = 0
    for i in range(len(change_of_supnet_pos) - 1):
        ax.hlines(change_of_supnet_pos[i], start, change_of_supnet_pos[i + 1], colors="violet", linewidth=2.0)
        ax.vlines(change_of_supnet_pos[i], start, change_of_supnet_pos[i + 1], colors="violet", linewidth=2.0)
        start = change_of_supnet_pos[i]
    ax.hlines(change_of_supnet_pos[-1], start, end, colors="violet", linewidth=2.0)
    ax.vlines(change_of_supnet_pos[-1], start, end, colors="violet", linewidth=2.0)

    label_pos = []
    start = 0
    for i in range(np.ceil(len(all_nets) / 2).astype(int)):
        label_pos.append(start + (change_of_net_pos[i] - start) / 2)
        start = change_of_net_pos[i]

    start = int(len(corr) / 2)
    for i in np.arange(np.ceil(len(all_supnets) / 2).astype(int), len(all_supnets)):
        if i != len(all_supnets) - 1:
            label_pos.append(start + (change_of_supnet_pos[i] - start) / 2)
            start = change_of_supnet_pos[i]
            continue
        label_pos.append(start + (end - start) / 2)

    x_labels = all_nets[:np.ceil(len(all_nets) / 2).astype(int)] + all_supnets[
                                                                    np.ceil(len(all_supnets) / 2).astype(int):]
    ax.set_xticks(label_pos)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_yticks([50, 150])
    ax.set_yticklabels(['LH', 'RH'], fontsize=8)
    output_path = os.path.join(out_dir, f"schaefer200_17networks_{'Neurochems_system_correlation_map'}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
        



#%% Paths 
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, '../'))
neurochems_path = os.path.join(parent_path, 'meg_outputs/schaefer200_17networks_neurochems_data.csv')
out_path = path2neurochem = os.path.join(parent_path, 'out_files')
#%%Fc Files and sorting index
#FcFile = np.sort(os.listdir(path2fc))

# read Neurochems

# Read the CSV file directly without storing the path in a separate variable
neurochems = pd.read_csv(neurochems_path)
labels = neurochems['region']
# Extract neurotrans/receptor names directly without creating an intermediate list
columns = neurochems.columns.tolist()

# Drop columns using iloc
neurochems.drop(columns=columns[0:2] + [columns[-1]], inplace=True)

#%% Compute correlation matrix directly using DataFrame's corr method

for key in neurochems.keys():
    print(key)
    neurochems_corr = neurochems.drop(columns=key).T.corr()
    neurochems_corr_df = ({'neurochems_corr': neurochems_corr,
                        'labels': labels})
    with open(os.path.join(parent_path, 'meg_outputs/schaefer200_17networks_neurochems~'+key+'_correlation_df.pkl'), "wb") as f:
            pkl.dump(neurochems_corr_df, f)

    # Plot heatmap using seaborn directly
    plot_neurocorr_matrix(neurochems_corr,labels,out_path)


# %%
