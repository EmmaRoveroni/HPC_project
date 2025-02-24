# Network analysis for TSPO PET data

import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TSPONet_functions import minmax_norm, euclidean_distance, cosine_similarity, compute_TSPONet, compute_node_strength
from scipy.io import savemat

folderData = '/nfsd/biopetmri4_tesi/BarzonLeonardo_2024/Neuroinflammation_PBR/data'
folderInfo = '/nfsd/biopetmri4_tesi/BarzonLeonardo_2024/Neuroinflammation_PK/info'
folderResults_KL = '/nfsd/biopetmri4/Users/LeonardoBarzon/TSPONet/results/Networks_KL'
folderResults_Eucl_Cos = '/nfsd/biopetmri4/Users/LeonardoBarzon/TSPONet/results/Networks_euclidean_cosine'

Group = [] 
compute_euclidean_cosine = True
compute_KL_distance = False
TSPONet_all = [] 

for idx, subj in enumerate(os.listdir(folderData)):
    if subj.startswith("SubjPBR_"):
        subj_path=os.path.join(folderData, subj)
        if os.path.isdir(subj_path):
            print(f"Working on {subj} ")

            # Load dynamic PET data
            dynPET_nifti = nib.load(f"/nfsd/biopetmri4_tesi/BarzonLeonardo_2024/Neuroinflammation_PBR/data/{subj}/PET/MNI_raip{subj}.hdr")
            dynPET = dynPET_nifti.get_fdata()

            # Load K1 map
            K1map_nifti = nib.load(f"/nfsd/biopetmri4_tesi/BarzonLeonardo_2024/Neuroinflammation_PBR/results/LogisticRegression/K1_maps/{subj}.hdr")
            K1map = K1map_nifti.get_fdata()

            # Load atlas in subject space
            Atlas_nifti = nib.load(f"/nfsd/biopetmri4_tesi/BarzonLeonardo_2024/Neuroinflammation_PBR/data/{subj}/SVCAmasks/aparc+aseg_2_PET.nii.gz") #Desikan 
            Atlas = Atlas_nifti.get_fdata()
            # Read labels
            SubCor_INFO = pd.read_excel(f"{folderInfo}/freesurfer/Labels_Desikan_Destrieux.xlsx", sheet_name="Subcorticals")
            SubCor_Names = SubCor_INFO['Name'].astype(str)  # Convert to string
            SubCor_Labels = SubCor_INFO['Label']
            Desikan_INFO = pd.read_excel(f"{folderInfo}/freesurfer/Labels_Desikan_Destrieux.xlsx", sheet_name="DesikanCortex")
            Desikan_Names = Desikan_INFO['Name'].astype(str)  # Convert to string
            Desikan_Labels = Desikan_INFO['Label']
            DK_labels = np.concatenate([SubCor_Labels.values, Desikan_Labels.values])

            # Selected labels (remember function range does not include the last extreme)
            labelsCerebellum = [8, 47]  # LH e RH
            labelsSubcorticals = list(range(10, 14)) + [17, 18] + [26, 28] + list(range(49, 55)) + [58, 60]  # LH e RH 19 and 55 insula added
            labelDesCortex = [label for label in list(range(1001, 1036)) + list(range(2001, 2036)) if label not in [1004, 2004]] # LH e RH
            labelsBrainStem = [16]
            selected_labels = labelsBrainStem + labelsCerebellum + labelsSubcorticals + labelDesCortex
            selected_labels = np.array(selected_labels)
           # selected_labels = np.sort(selected_labels)
            n_roi = len(selected_labels)

            # Select time points based on logistic regression analysis
            sel = [6, 16, 23]
            TAC_selected = [dynPET[:, :, :, idx] for idx in sel] # NOrmalize TAC by dose/weight
            info = pd.read_excel(f'/nfsd/biopetmri4_tesi/BarzonLeonardo_2024/Neuroinflammation_PBR/info/SubjectsInfo_anonymization.xlsx')
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
            n_features = Features.shape[3]

            if n_features > 1 and compute_euclidean_cosine:

                # Initialize the output array for median features
                median_features = np.zeros((n_roi, n_features))

                # Compute the median features
                for i, current_roi in enumerate(selected_labels):
                    roi_mask = Atlas == current_roi                 # Create a mask for the current ROI
                    if np.any(roi_mask):                            # Verify if the ROI is included
                        for f in range(n_features):
                            current_feat = Features[..., f]             # Extract the current feature map
                            tmp = current_feat[roi_mask]                # Extract values for the current ROI
                            median_features[i, f] = np.nanmedian(tmp)      # Compute the median and store it
                    else:
                        median_features[i,:] = np.nan 

                # Compute distance between ROIs
                eucl_dist = np.zeros((n_roi, n_roi))
                cos_sim = np.zeros((n_roi, n_roi))
                KL_dist = np.zeros((n_roi, n_roi))
                for i in range(n_roi):
                    for j in range(n_roi):
                        if (i != j) and not (np.isnan(median_features[i, :]).any() or np.isnan(median_features[j, :]).any()):
                            # Compute Euclidean distance between ROI feature vectors
                            fi = minmax_norm(median_features[i,:])
                            fj = minmax_norm(median_features[j,:])
                            eucl_dist[i, j] = euclidean_distance(fi, fj)
                            cos_sim[i, j] = 1 - cosine_similarity(fi, fj)
                        else:
                            eucl_dist[i,j] = np.nan
                            cos_sim[i,j] = np.nan

                # Normalize distances to obtain Euclidean similarity
                max_distance = np.nanmax(eucl_dist)  # Maximum distance for the subject
                max_cos = np. nanmax(cos_sim)

                if max_distance > 0:
                    sim_matrix = 1 - (eucl_dist / max_distance)
                else:
                    sim_matrix = np.ones_like(eucl_dist)  # Handle case where max_distance = 0

                if max_cos > 0:
                    cos_matrix = 1 - (cos_sim / max_cos)
                else:
                    cos_matrix = np.ones_like(cos_sim)  # Handle case where max_cos = 0

                # Apply Fisher z-transformation to stabilize variance
                fisher_z_sim = np.arctanh(sim_matrix)  # Fisher z-transformation
                fisher_z_cos = np.arctanh(cos_matrix)

                # Rescale Fisher z-transformed values back to [0, 1]
                min_z = np.nanmin(fisher_z_sim)
                max_z = np.nanmax(fisher_z_sim)
                if max_z > min_z:  # Avoid division by zero
                    rescaled_sim_matrix = (fisher_z_sim - min_z) / (max_z - min_z)
                else:
                    rescaled_sim_matrix = np.zeros_like(fisher_z_sim)  # Handle edge case

                min_z = np.nanmin(fisher_z_cos)
                max_z = np.nanmax(fisher_z_cos)
                if max_z > min_z:  # Avoid division by zero
                    rescaled_cos_matrix = (fisher_z_cos - min_z) / (max_z - min_z)
                else:
                    rescaled_cos_matrix = np.zeros_like(fisher_z_cos)  # Handle edge case

                file_name = os.path.join(folderResults_Eucl_Cos, f"{subj}.mat")
                savemat(file_name,{"Net_euclidean" : rescaled_sim_matrix, "Net_cosine": rescaled_cos_matrix})
                # # Create two subplots (1 row, 2 columns)
                # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                
                # # Plot the rescaled similarity matrix
                # ax1 = axes[0]  # First subplot
                # cax1 = ax1.imshow(rescaled_sim_matrix, cmap='jet', interpolation='nearest')
                # fig.colorbar(cax1, ax=ax1)  # Add colorbar
                # ax1.set_title("Rescaled Euclidean Similarity Matrix")
                # ax1.set_xlabel("ROI")
                # ax1.set_ylabel("ROI")

                # # Plot the rescaled cosine similarity matrix
                # ax2 = axes[1]  # Second subplot
                # cax2 = ax2.imshow(rescaled_cos_matrix, cmap='jet', interpolation='nearest')
                # fig.colorbar(cax2, ax=ax2)  # Add colorbar
                # ax2.set_title("Rescaled Cosine Similarity Matrix")
                # ax2.set_xlabel("ROI")
                # ax2.set_ylabel("ROI")



            ## Similarity based on ROI distributions
            if compute_KL_distance:
                TSPONet_dist = compute_TSPONet(Features, Atlas, selected_labels, standardize_features=True, visual_report=False, visualize_results=False) 
                file_name = os.path.join(folderResults_KL, f"{subj}.mat")
                savemat(file_name,{"TSPONet" : TSPONet_dist})
                TSPONet_all.append(TSPONet_dist)

            # Compute node strength
           # node_strength = compute_node_strength(TSPONet_dist)

            # plt.figure(figsize=(10, 6))
            # x_values = np.arange(len(selected_labels))  # Numeric vector for the X-axis
            # plt.bar(x_values, node_strength, color='skyblue', edgecolor='black')
            # plt.title("Node Strength per Brain Region", fontsize=16)
            # plt.xlabel("Brain Regions", fontsize=12)
            # plt.ylabel("Node Strength", fontsize=12)
            # plt.xticks(x_values, selected_labels, rotation=45, ha='right', fontsize=10) 
            # plt.tight_layout()  
            # plt.show()

# savemat(os.path.join(folderResults, "Matrices_all.mat"), {"Matrices" : TSPONet_all})
