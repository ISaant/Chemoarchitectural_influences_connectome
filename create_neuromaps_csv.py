#%%
import pandas as pd
import numpy as np
from neuromaps import datasets, parcellate, transforms
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy import stats
import warnings
warnings.simplefilter("ignore")

# Load the Schaefer200 parcellation with 17 Networks
schaefer = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17)
parcellater = parcellate.Parcellater(schaefer['maps'], 'MNI152')
neuromaps_detailed_table_url = "https://docs.google.com/spreadsheets/d/1oZecOsvtQEh5pQkIf8cB6CyhPKVrQuko/gviz/tq?tqx=out:csv&gid=1162991686"
neuromaps_detailed_table = pd.read_csv(neuromaps_detailed_table_url)

# Function to extract mean age:

def mean_age(age):

    if " " in age and len(age)>6: 
        m_age = np.float64((age[:6].split(' ')[0])) 
    elif "-" in age:
        m_age = np.mean([int(x) for x in age[:5].split("-")])
    return m_age

# Define relevant neurotransmitter systems

neurotransmitter_systems = {
    '5HT1a':['cumi101', 'way100635'],
    '5HT1b':['az10419369','p943'],
    '5HT2a':['cimbi36', 'altanserin'],
    '5HT4':['sb207145'],
    '5HT6':['gsk215083'],
    '5HTT':['dasb','madam'],
    'a4b2':['flubatine'],
    'CB1':['fmpepd2', 'omar'],
    'D1': ['sch23390'],
    'D2': ['raclopride', 'flb457', 'fallypride'],
    'DAT': ['fpcit', 'fepe2i'],
    'GABAa': ['flumazenil', 'ro154513'],
    'H3':['gsk189254'],
    'M1': ['lsn3172176'],
    'mGluR5': ['abp688'],
    'MOR': ['carfentanil'],
    'NET': [ 'mrb', 'methylreboxetine'],
    'NMDA': ['ge179',],
    'VAChT':['feobv']
}



# Get the list of available neurotransmitter annotations
annotations = datasets.available_annotations()
annotations.remove(('castrillon2023', 'cmrglc', 'MNI152', '3mm'))
neurotransmitter_data = {}
neurotransmitter_age = {}

sources = [source for source, desc, space, res in annotations]
descs = [desc for source, desc, space, res in annotations]
spaces = [space for source, desc, space, res in annotations]
reses = [res for source, desc, space, res in annotations]
#%%

#seen_descs = []


for system, ligands in neurotransmitter_systems.items():
    print(system)
    N=0
    parcellated = []
    m_age = []
    for ligand in ligands:
        indx = np.argwhere(np.logical_and(np.array(spaces)=='MNI152',np.array(descs)==ligand)).flatten()
        if indx.size > 1:
            for idx in indx:
                annotation = datasets.fetch_annotation(source=sources[idx], desc=descs[idx], space='MNI152',verbose=0)
                n = np.array(neuromaps_detailed_table.iloc[[idx]]['N (males)'].astype(str).str.split(' ').str[0]).astype(int)[0]
                age = mean_age(neuromaps_detailed_table.iloc[[idx]]['age (years)'].iloc[0])
                print(idx,n, age)
                parcellated.append(stats.zscore(parcellater.fit_transform(annotation, 'MNI152').flatten())*n)
                m_age.append(age*n)
                N += n
            
        else:
            annotation = datasets.fetch_annotation(source=sources[indx[0]], desc=descs[indx[0]], space='MNI152', verbose=0)
            n = np.array(neuromaps_detailed_table.iloc[indx]['N (males)'].astype(str).str.split(' ').str[0]).astype(int)[0]
            age = mean_age(neuromaps_detailed_table.iloc[indx]['age (years)'].iloc[0])
            print(indx,n,age)
            parcellated.append(stats.zscore(parcellater.fit_transform(annotation, 'MNI152').flatten())*n)
            m_age.append(age*n)
            N += n
    parcellated = np.sum(parcellated,axis=0)/N   
    m_age = np.sum(m_age)/N
    print(f'mean:{np.mean(parcellated)}, max:{np.max(parcellated)}, min:{np.min(parcellated)}')
    print(f'{system} mean age: {m_age}')
    # Ensure data is 1D before adding it to the dictionary with the system label
    neurotransmitter_data[system] = parcellated.flatten()
    neurotransmitter_age[system] = m_age
    #%% Extract region names from the parcellation
regions = [label.decode('utf-8') for label in schaefer['labels']]
hemispheres = ['lh' if 'LH' in label else 'rh' for label in regions]

# Create a DataFrame with neurotransmitter data
df_neurotransmitters = pd.DataFrame(neurotransmitter_data)

# Add metadata columns
df_neurotransmitters.insert(0, 'region', regions)  # Insert the region column at the beginning
df_neurotransmitters.insert(1, 'hemi', hemispheres)  # Insert the hemisphere column

df_neurotransmitters['scale'] = 'Schaefer200'
df_neurotransmitters = df_neurotransmitters.sort_values(by="region")        
df_age = pd.DataFrame(neurotransmitter_age.items(), columns = ['system', 'mean age'])

#%% Save as CSV
df_neurotransmitters.to_csv("/Users/isaant/Documents/PosDoc/Projects/Shaping_aging_fc/meg_outputs/schaefer200_17networks_neurochems_data.csv", index=False)
df_age.to_csv("/Users/isaant/Documents/PosDoc/Projects/Shaping_aging_fc/meg_outputs/schaefer200_17networks_neurochems_age.csv", index=False)

print("CSV file generated with Schaefer200 17 Networks parcellation and neurotransmitter maps labeled by system.")

# %%
