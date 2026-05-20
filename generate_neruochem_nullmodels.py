#%%


import os
import pandas as pd
import numpy as np
import nibabel as nib
import nibabel.freesurfer.io as fsio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from neuromaps.nulls import hungarian
from neuromaps.images import annot_to_gifti, relabel_gifti
from nilearn.datasets import fetch_atlas_schaefer_2018
from netneurotools.datasets import fetch_schaefer2018
from nibabel.freesurfer.io import write_annot
from nilearn import plotting, datasets
from collections import namedtuple
from pathlib import Path


### Functions 

def reorder_annot(annot_path, region_order):
    # Load the .annot file
    labels, ctab, names = fsio.read_annot(annot_path)
    names_str = [n.decode('utf-8') for n in names]

    # Create a mapping from region name to original index
    name_to_index = {name: i for i, name in enumerate(names_str)}

    # Match and sort according to the desired region order
    sorted_indices = [name_to_index[r] for r in region_order if r in name_to_index]

    # Reorder names and color table
    names_sorted = np.array(names)[sorted_indices]
    ctab_sorted = ctab[sorted_indices]

    # Build remapping: old index -> new index
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}

    # Remap the label indices
    labels_remapped = np.array([old_to_new.get(l, l) for l in labels])

    return labels_remapped, ctab_sorted, names_sorted

def compare_annots(original_annot, reordered_annot, hemi='left'):
    print(f"\n🔍 Comparing hemisphere: {hemi.upper()}")

    # Load both .annot files
    labels_orig, ctab_orig, names_orig = fsio.read_annot(original_annot)
    labels_new, ctab_new, names_new = fsio.read_annot(reordered_annot)

    # Decode region names
    names_orig = [n.decode('utf-8') for n in names_orig]
    names_new = [n.decode('utf-8') for n in names_new]

    # Check if all region names are preserved
    missing = set(names_orig) - set(names_new)
    added = set(names_new) - set(names_orig)
    if missing:
        print("Missing regions in reordered annot:", missing)
    if added:
        print("Extra regions in reordered annot:", added)
    if not missing and not added:
        print("All region names are preserved.")

    # Check if region colors match
    name_to_color_orig = {name: ctab_orig[i][:3] for i, name in enumerate(names_orig)}
    name_to_color_new = {name: ctab_new[i][:3] for i, name in enumerate(names_new)}

    mismatched_colors = []
    for name in name_to_color_orig:
        if name in name_to_color_new:
            if not np.array_equal(name_to_color_orig[name], name_to_color_new[name]):
                mismatched_colors.append(name)

    if mismatched_colors:
        print("Regions with mismatched colors:", mismatched_colors)
    else:
        print("All region colors match.")

    # Vertex-wise sample check
    print("Vertex comparison (sample):")
    for v in [100, 1000, 5000, 10000]:
        l_orig = labels_orig[v]
        l_new = labels_new[v]
        name_orig = names_orig[l_orig]
        name_new = names_new[l_new]
        print(f"Vertex {v}: Original = {name_orig}, New = {name_new}")

    # Visualization
    print("Plotting original vs. reordered labels...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    surf_file = fsaverage.pial_left if hemi == 'left' else fsaverage.pial_right

    plotting.plot_surf_roi(surf_file, roi_map=labels_orig, hemi=hemi,
                           title=f"{hemi.upper()} - Original Annot")
    plotting.plot_surf_roi(surf_file, roi_map=labels_new, hemi=hemi,
                       title=f"{hemi.upper()} - Reordered Annot")
    
    # Visualize surface using true FreeSurfer ctab colors
    print("Plotting using FreeSurfer ctab colors...")
    gii = nib.load(surf_file)
    coords = gii.darrays[0].data
    faces = gii.darrays[1].data

    def plot_annot_surface(coords, faces, labels, ctab, title):
        rgb_ctab = ctab[:, :3] / 255.
        vertex_colors = rgb_ctab[labels]

        # Color por triángulo: promedio de los colores de los vértices de cada cara
        face_colors = np.mean(vertex_colors[faces], axis=1)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(coords[faces], facecolors=face_colors, linewidths=0.05)
        ax.add_collection3d(mesh)

        ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
        ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
        ax.set_zlim(coords[:, 2].min(), coords[:, 2].max())
        ax.view_init(elev=90, azim=0)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    plot_annot_surface(coords, faces, labels_orig, ctab_orig, f"{hemi.upper()} - Original Annot (FreeSurfer Colors)")
    plot_annot_surface(coords, faces, labels_new, ctab_new, f"{hemi.upper()} - Reordered Annot (FreeSurfer Colors)")




### Paths:
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, '../'))
neurochems_path = os.path.join(parent_path, 'meg_outputs/schaefer200_17networks_neurochems_data.csv')

### Download schaefer 200 17 Netwroks parcellation map 
#schaefer = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17) # This map is in MNI152 (volumetric)
schaefer_parcels_fsaverage = fetch_schaefer2018('fsaverage')['200Parcels17Networks'] #This map is in fsaverage (surface)

# Read LH .annot files
#An .annot file associates each vertex of a hemisphere's surface with:
#   - A label (index)
#   - A region name
#   - An RGBA color for display

#lhannot = schaefer_parcels_fsaverage[0]
#labels, ctab, names = fsio.read_annot(lhannot)
#print(names)

### Reorder annot to match neuromaps order of regions.
neurochems = pd.read_csv(neurochems_path)
region_order = neurochems['region'].tolist()
lh_regions = [r for r in region_order if '_LH_' in r]
rh_regions = [r for r in region_order if '_RH_' in r]

# Reorder and save new .annot files
lh_labels, lh_ctab, lh_names = reorder_annot(schaefer_parcels_fsaverage[0], lh_regions)
rh_labels, rh_ctab, rh_names = reorder_annot(schaefer_parcels_fsaverage[1], rh_regions)

#Write the .annot files
write_annot(os.path.join(parent_path,'schaefer20017Networks_annot_reordered',
                         'lh_Schaefer2018_200Parcels_17Networks_reordered.annot'),
                           lh_labels, lh_ctab, lh_names)
write_annot(os.path.join(parent_path,'schaefer20017Networks_annot_reordered',
                         'rh_Schaefer2018_200Parcels_17Networks_reordered.annot'),
                           rh_labels, rh_ctab, rh_names)

# Compare and visualize both hemispheres
compare_annots(schaefer_parcels_fsaverage[0],
               os.path.join(parent_path,'schaefer20017Networks_annot_reordered',
                         'lh_Schaefer2018_200Parcels_17Networks_reordered.annot'), hemi='left')

compare_annots(schaefer_parcels_fsaverage[1],
               os.path.join(parent_path,'schaefer20017Networks_annot_reordered',
                         'rh_Schaefer2018_200Parcels_17Networks_reordered.annot'), hemi='right')



#%% 
### Transform from annot to Gifti format

# GIfTI is the Geometry format under the Neuroimaging Informatics Technology Initiative (NIfTI).
# Basically, it is the surface-file format complement to the NIfTI volume-file format
# NIFTI (Neuroimaging Informatics Technology Initiative) volume format is 
#  a standard file format for storing volumetric neuroimaging data. 

Surface = namedtuple('Surface', ['L', 'R'])
schaefer_parcels_fsaverage_reordered = Surface(
    L=Path(os.path.join(parent_path,'schaefer20017Networks_annot_reordered',
                         'lh_Schaefer2018_200Parcels_17Networks_reordered.annot')),
    R=Path(os.path.join(parent_path,'schaefer20017Networks_annot_reordered',
                         'rh_Schaefer2018_200Parcels_17Networks_reordered.annot'))
)
schaefer_parcels_fsaverage_reordered = annot_to_gifti (schaefer_parcels_fsaverage_reordered)
# schaefer_parcels_fsaverage = relabel_gifti (schaefer_parcels_fsaverage) 
lh,rh = schaefer_parcels_fsaverage_reordered
lh_label = [label.label for label in lh.labeltable.labels]
rh_label = [label.label for label in rh.labeltable.labels]

#for lh_label, rh_label in zip(lh.labeltable.labels,rh.labeltable.labels):
#    print(lh_label.label, rh_label.label)

neurochems = pd.read_csv(neurochems_path)

#Here we compare if the order of the labels are the same! 
comparing_labels1 = pd.DataFrame({'labels': neurochems['region']})
comparing_labels2 = pd.DataFrame({'labels': lh_label + rh_label})
print(comparing_labels1.compare(comparing_labels2)) # If equal, the print should be empty
#%%Fc Files and sorting index
#FcFile = np.sort(os.listdir(path2fc))

# read Neurochems

# Read the CSV file directly without storing the path in a separate variable
# Inner wall (Background+FreeSurfer_Defined_Medial_Wall) is excluded via PARCIGNORE in get_parcel_centroids;
#  remaining parcels are projected onto a sphere and rotated 5,000 times (or as specified).
# Call inspect.getsource(hungarian), inspect.getsource(get_parcel_centroids), neuromaps.images.PARCIGNORE 
#  for better understanding. inspect is a library
# schaefer_parcels_fsaverage = relabel_gifti (schaefer_parcels_fsaverage) 
null_models_hung = hungarian (data=None, atlas='fsaverage', density='164k', #Density of surface mesh on which data are defined. Must be compatible with specified atlas. 'maps': '/Users/isaant/nilearn_data/schaefer_2018/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                                parcellation=schaefer_parcels_fsaverage_reordered, n_perm=1000, seed=1, 
                                spins=None, surfaces=None)

regions_and_index = pd.DataFrame({'original_index': np.arange(200),
                                  'labels': lh_label + rh_label}) 

# %%
out_path='/Users/isaant/Documents/PosDoc/Projects/Shaping_aging_fc/meg_outputs'
#To save as csv instead of pickle for easier manipulation in R
df = pd.DataFrame(null_models_hung)
df.to_csv(os.path.join(out_path,'null_model_hung_index.csv'), index=False)

import pickle

with open(os.path.join(out_path,'null_models_hung.pkl'), 'wb') as handle:
    pickle.dump(null_models_hung, handle)

with open(os.path.join(out_path,'regions_and_index_hungSpin.pkl'), 'wb') as handle:
    pickle.dump(regions_and_index, handle)
# %%
