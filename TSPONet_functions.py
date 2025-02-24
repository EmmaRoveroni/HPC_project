# TSPONet framework functions
from scipy.spatial import cKDTree as KDTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def cosine_similarity(vec1, vec2):
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    # Compute magnitudes of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # Handle edge case to avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return np.nan
    # Compute cosine similarity
    return dot_product / (norm_vec1 * norm_vec2)

def minmax_norm(vec):
    return (vec- np.min(vec)) / (np.max(vec) - np.min(vec))

def get_KDTree(x): #Inspired by https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    
    # Build a KD tree representation of the samples
    xtree = KDTree(x)
    
    return xtree

def get_KL(x, y, xtree, ytree): #Inspired by https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape
    
    #Check dimensions
    assert(d == dy)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]
    
    rs_ratio = r/s

    #Remove points with zero, nan, or infinity. This happens when two regions have a vertex with the exact same value – an occurence that basically onnly happens for the single feature MSNs
    #and has to do with FreeSurfer occasionally outputting the exact same value for different vertices.
    rs_ratio = rs_ratio[np.isfinite(rs_ratio)]
    rs_ratio = rs_ratio[rs_ratio!=0.0]
    
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.

    kl = -np.log(rs_ratio).sum() * d / n + np.log(m / (n - 1.))
    kl = np.maximum(kl, 0)
    
    return kl



def compute_TSPONet(Features, Atlas, selected_ROIs, standardize_features, visual_report, visualize_results):
   # computes TSPONet using KL divergence
   # INPUTs:
   # - Features: a 4D numpy array composed by stacked 3D maps, the 4th dimension is the number of features
   # - Atlas: 3D numpy array; tha atlas must be coregistered to the maps provided in Features
   # - selected_ROIs: labels with the selected regions from the atlas 
   # - standardize_features: set to True if you want to standardize features across all the brain, False otherwise
   # - visual_report: set to True if you want to visualize the intermediate results
   # - visualize_network: se to True if you want to visualize the final similarity matrix


    n_roi=len(selected_ROIs)
    if Features.ndim == 4:
        n_features = Features.shape[3]
    elif Features.ndim ==3:
        n_features = 1
    else:
        raise(Exception("Error: Bad formatted array of features in input. Features must be a 3D (1 feature) or 4D (n feature) numpy array)"))

    TSPONet_dist = np.zeros((n_roi, n_roi))

    # Normalize features across all the brain using robust scaling (median, iqr)
    if standardize_features:
        Atlas_mask = Atlas > 0
        if n_features ==1:
            m = np.nanmedian(Features[Atlas_mask])
            iqr = np.nanpercentile(Features[Atlas_mask], 75) - np.nanpercentile(Features[Atlas_mask], 25)
            Features_std = (Features - m) / iqr
        else:
           # print(Features.shape)
            Features_std = np.empty(Features.shape)
            for f in range(n_features):
                m = np.nanmedian(Features[Atlas_mask, f])
                iqr = np.nanpercentile(Features[Atlas_mask, f], 75) - np.nanpercentile(Features[Atlas_mask, f], 25)
                Features_std[Atlas_mask, f] = (Features[Atlas_mask, f] - m) / iqr
           # print(Features_std.shape)

    for i,roi_i in enumerate(selected_ROIs):
        for j, roi_j in enumerate(selected_ROIs):
            if (i < j):
                print(f"\rTSPONet: Computing ROIs {i} and {j}...", end="")
                roi_mask_i = Atlas == roi_i
                roi_mask_j = Atlas == roi_j
                length_roi_i = np.sum(roi_mask_i)  
                length_roi_j = np.sum(roi_mask_j)
                if length_roi_i!=0 and length_roi_j!=0:
                    data_i = np.empty((length_roi_i,0))
                    data_j = np.empty((length_roi_j,0))
                    if n_features == 1:
                        current_feat = Features_std
                        data_i = current_feat[roi_mask_i].reshape(-1,1)       # Extract values for the current ROI
                        data_j = current_feat[roi_mask_j].reshape(-1,1)
                        #Check that there aren't many repeated values
                        percent_unique_vals_i = len(np.unique(data_i, axis=0))/len(data_i)
                        percent_unique_vals_j = len(np.unique(data_j, axis=0))/len(data_j)
                        if percent_unique_vals_i < 0.8 or percent_unique_vals_j < 0.8:
                            print("Warning: There are many repeated values in the data, which compromises the validity of TSPONet calculation. Since you are using only 1 feature, try to resample the data")
                    else:
                        for f in range(n_features):
                            current_feat = Features_std[..., f]                      # Extract the current feature map
                            tmp_i = current_feat[roi_mask_i].reshape(-1,1)       # Extract values for the current ROI
                            tmp_j = current_feat[roi_mask_j].reshape(-1,1)
                            percent_unique_vals_i = len(np.unique(tmp_i))/len(tmp_i)
                            percent_unique_vals_j = len(np.unique(tmp_j))/len(tmp_j)
                            if percent_unique_vals_i < 0.8 or percent_unique_vals_j < 0.8:
                                 print(f"Warning: There are many repeated values in feature {f}  , which compromises the validity of TSPONet calculation")
                            data_i = np.concatenate((data_i, tmp_i), axis=1)
                            data_j = np.concatenate((data_j, tmp_j), axis=1)
                    KLa = get_KL(data_i, data_j, get_KDTree(data_i), get_KDTree(data_j))   # compute KL divergence 
                    KLb = get_KL(data_j, data_i, get_KDTree(data_j), get_KDTree(data_i))   
                    Kl = KLa + KLb                                                         # Symmetric KL divergence 
                    TSPONet_dist[i,j] =  1 / (1 + Kl)                                      # similarity metric  
                    TSPONet_dist[j,i] =  1 / (1 + Kl)    
                else:
                     TSPONet_dist[i,j] = np.nan
                     TSPONet_dist[j,i] = np.nan
            elif i == j:
                TSPONet_dist[i,j] = 1 #because KL = 0 
    
    if visual_report:
        for f in range(n_features):
            feature_original = Features[..., f] if n_features > 1 else Features
            atlas_mask = np.isin(Atlas, selected_ROIs)
            feature_masked = np.where(atlas_mask, feature_original, np.nan)
            if standardize_features:
                feature_standardized = np.where(atlas_mask, Features_std[..., f] if n_features > 1 else Features_std, np.nan)
                n_subplots = 3
            else:
                n_subplots = 2

            # Create the figure
            fig, axes = plt.subplots(1, n_subplots, figsize=(18, 6))

            im0 = axes[0].imshow(np.rot90(feature_original[:, :, feature_original.shape[2] // 2]), cmap='hot')
            axes[0].set_title(f'Original Feature {f+1}')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(np.rot90(feature_masked[:, :, feature_masked.shape[2] // 2]), cmap='hot')
            axes[1].set_title(f'Masked Feature {f+1}')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])

            if standardize_features:
                im2 = axes[2].imshow(np.rot90(feature_standardized[:, :, feature_standardized.shape[2] // 2]), cmap='hot')
                axes[2].set_title(f'Standardized Feature {f+1}')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2])

            # Show the figure
            plt.title(f"Feature {f}")
            plt.tight_layout()
            plt.show()
        
    if visualize_results:

        fig, ax = plt.subplots(figsize=(8, 6))  
        cax = ax.imshow(TSPONet_dist, cmap='jet', interpolation='nearest')
        fig.colorbar(cax, ax=ax, label="Similarity (1 / (1 + KL))")

        ax.set_title("TSPONet Similarity Matrix", fontsize=14)
        ax.set_xlabel("ROI", fontsize=12)
        ax.set_ylabel("ROI", fontsize=12)

        ax.set_xticks(range(len(selected_ROIs)))
        ax.set_yticks(range(len(selected_ROIs)))
        ax.set_xticklabels(selected_ROIs, rotation=90, fontsize=10) 
        ax.set_yticklabels(selected_ROIs, fontsize=10)

        plt.tight_layout()
        plt.show()
    print("\n")
    return TSPONet_dist


def compute_node_strength(Network_matrix):
   # Computes node strength of the network provided in input
   # Input:
   # - Network_matrix: squared 2D numpy array representing weights of connection between nodes
   # Output:
   # - Node strenght of each region (node)

    if Network_matrix.shape[0] != Network_matrix.shape[1]:
        raise Exception("Input must be a square matrix")  
   
   
    n_roi = Network_matrix.shape[0]
    np.fill_diagonal(Network_matrix, np.nan)
    node_strength = np.nansum(Network_matrix, axis=1)

    return node_strength