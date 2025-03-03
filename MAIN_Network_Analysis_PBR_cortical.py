import os
import nibabel as nib
import numpy as np
import pandas as pd
import argparse
from TSPONet_functions import compute_TSPONet
from scipy.io import savemat

def main():
    parser = argparse.ArgumentParser(description='Network analysis for TSPO PET data')
    parser.add_argument('--folderData', required=True, help='Path to data folder')
    parser.add_argument('--folderAtlasInfo', required=True, help='Path atals info folder')
    parser.add_argument('--folderSubjectsInfo', required=True, help='Path subjects info folder')
    parser.add_argument('--folderResults', required=True, help='Path where to store results')
    args = parser.parse_args()
    
    folderData = args.folderData
    folderAtlasInfo = args.folderAtlasInfo
    folderSubjectsInfo = args.folderSubjectsInfo
    folderResults = args.folderResults

    Group = [] 

    TSPONet_all = [] 
    count = 0
    for idx, subj in enumerate(os.listdir(folderData)):
        if subj.startswith("Subj") and count<100:
            subj_path=os.path.join(folderData, subj)
            if os.path.isdir(subj_path):
                count = count + 1
                print(f"Working on {subj} ")

                # Load dynamic PET data
                dynPET_nifti = nib.load(os.path.join(folderData, subj, "PET", f"MNI_raip{subj}.hdr"))
                dynPET = dynPET_nifti.get_fdata()

                # Load K1 map
                K1map_nifti = nib.load(os.path.join(folderData, "../results", "LogisticRegression", "K1_maps", f"{subj}.hdr"))
                K1map = K1map_nifti.get_fdata()

                # Load atlas in subject space
                Atlas_nifti = nib.load(os.path.join(folderData, subj, "SVCAmasks", "aparc+aseg_2_PET.nii.gz"))
                Atlas = Atlas_nifti.get_fdata()

                # Read labels
                SubCor_INFO = pd.read_excel(os.path.join(folderAtlasInfo, "freesurfer", "Labels_Desikan_Destrieux.xlsx"), sheet_name="Subcorticals")
                SubCor_Names = SubCor_INFO['Name'].astype(str)  # Convert to string
                SubCor_Labels = SubCor_INFO['Label']
                Desikan_INFO = pd.read_excel(os.path.join(folderAtlasInfo, "freesurfer", "Labels_Desikan_Destrieux.xlsx"), sheet_name="DesikanCortex")
                Desikan_Names = Desikan_INFO['Name'].astype(str)  # Convert to string
                Desikan_Labels = Desikan_INFO['Label']
                DK_labels = np.concatenate([SubCor_Labels.values, Desikan_Labels.values])

                # Selected labels
                # labelsCerebellum = [8, 47]  # LH e RH
                # labelsSubcorticals = list(range(10, 14)) + [17, 18] + [26, 28] + list(range(49, 55)) + [58, 60]
                labelDesCortex = [label for label in list(range(1001, 1036)) + list(range(2001, 2036)) if label not in [1004, 2004]]
                # labelsBrainStem = [16]
                # selected_labels = np.array(labelsBrainStem + labelsCerebellum + labelsSubcorticals + labelDesCortex)
                selected_labels = np.array(labelDesCortex)

                # Select time points
                sel = [6, 16, 23]
                TAC_selected = [dynPET[:, :, :, idx] for idx in sel]
                info = pd.read_excel(os.path.join(folderSubjectsInfo, "SubjectsInfo_anonymization.xlsx"))
                Subject_info = info[info['SubjID'] == subj]
                Group.append(Subject_info['Group'])  
                Dose = float(Subject_info['Dose'].iloc[0])
                Weight = float(Subject_info['Weight'].iloc[0])
                DW = Dose / Weight
                TAC_1 = TAC_selected[0] / DW
                TAC_2 = TAC_selected[1] / DW
                TAC_3 = TAC_selected[2] / DW

                # Stack the features into a 4D array
                Features = np.stack((TAC_1, TAC_2, TAC_3, K1map), axis=3)

                ## Similarity based on ROI distributions
                TSPONet_dist = compute_TSPONet(Features, Atlas, selected_labels, standardize_features=True, visual_report=False, visualize_results=False) 
                file_name = os.path.join(folderResults, f"{subj}.mat")
                savemat(file_name,{"TSPONet" : TSPONet_dist})
                TSPONet_all.append(TSPONet_dist)

    savemat(os.path.join(folderResults, "Matrices_HPC_project.mat"), {"Matrices" : TSPONet_all})

if __name__ == "__main__":
    main()