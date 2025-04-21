#functions for working with fMRI data

import os
import numpy as np
import pandas as pd
# import ipdb
import re
import pickle
from aux_functions import extract_id

# folder_path = '/content/s3bucket/data/Projects/ABIDE/Outputs/cpac/filt_noglobal/rois_cc200' #these hold .1D time course files

def fc_invalid(folder_path):
  if not os.path.exists(folder_path):
    print(f"Folder {folder_path} does not exist.")
    return []

  invalid_files = []

  for f in os.listdir(folder_path):
    f_path = os.path.join(folder_path, f)
    # check if .1D file
    if not f.endswith('.1D'):
      print(f"invalid file format: {f}")
      invalid_files.append(f)
      continue

    try:
      data = np.loadtxt(f_path, dtype=np.float32)
      df = pd.DataFrame(data)
      if df.isna().any().any():
        print("missing values detected in fMRI time series", f)
        missing_rois = df.isna().sum(axis=1)
        print("ROIs with missing values:", df.index[missing_rois].tolist())
        invalid_files.append(f)
    except Exception as e:
      print(f"error in loading file {f}: {e}")
      invalid_files.append(f)
      
  for i in range(len(invalid_files)):
    invalid_files[i] = extract_id(invalid_files[i])

  return invalid_files



def get_avg_corr_matrix(folder_path, invalid_files):
  # avg_corr_matrix = pd.DataFrame(0.0, index = range(200), columns = range(200))
  avg_corr_matrix = np.zeros((200, 200))
  file_count = np.zeros((200, 200))


  for f in os.listdir(folder_path):
    if f in invalid_files: #skip that file
      continue
    mask = np.ones((200, 200), dtype=bool) #to deal with nan correlation values
    f_path = os.path.join(folder_path, f)
    data = np.loadtxt(f_path,dtype=np.float32)
    # ipdb.set_trace()
    df = pd.DataFrame(data)
    if df.isna().any().any():
      print("Missing values detected in fMRI time series!", f)
      missing_rois = df.isna().sum(axis=1)
      print("ROIs with missing values:", df.index[missing_rois].tolist())
      break
    # calc the pearson correlation-matrix for each .1D file
    # ipdb.set_trace()
    correlation_matrix = df.corr(method='pearson') #now, correlation_matrix is not a df but a np array
    nan_indices = correlation_matrix.isna().stack()[lambda x: x].index.tolist() # eg- [(5,10), (6,11), ...]
    if len(nan_indices) > 0:
      ipdb.set_trace()
      for i,j in nan_indices:
        mask[i,j] = False

    file_count[mask] += 1
    # if np.isnan(correlation_matrix).any():
    #   print(correlation_matrix)
    #   break
    correlation_matrix = correlation_matrix.fillna(0).values
    avg_corr_matrix += correlation_matrix

    # for i in range(200):
    #   for j in range(i,200):
    #     # ipdb.set_trace()
    #     #building the avg. correlation matrix
    #     avg_corr_matrix.loc[i,j] += correlation_matrix.loc[i,j]

  avg_corr_matrix = avg_corr_matrix/file_count #this is the final avg_corr_matrix
  
  return avg_corr_matrix

  #now based on this, select extreme corr values from all the corr_matrices
    
def trim_corr_mat(mat):
  upper_tri_indices = np.triu_indices_from(mat, k=1)
  upper_tri_values = mat[upper_tri_indices]  # shape: (19900,)
  num_extreme = len(upper_tri_values) // 4  # top 25% and bottom 25%

  sorted_indices = np.argsort(upper_tri_values)
  # bottom 25%
  bottom_indices_flat = sorted_indices[:num_extreme]
  # top 25%
  top_indices_flat = sorted_indices[-num_extreme:]
  extreme_indices_flat = np.concatenate((bottom_indices_flat, top_indices_flat))
  extreme_coords = (upper_tri_indices[0][extreme_indices_flat],
                  upper_tri_indices[1][extreme_indices_flat])
  
  extreme_pairs = list(zip(extreme_coords[0], extreme_coords[1]))
  return extreme_pairs

def get_fc_feature_vector_compressed(invalid_files, pheno_invalid, extreme_pairs):
  features = {}

  for f in os.listdir(folder_path):
    subject_features = []
    if extract_id(f) in invalid_files or extract_id(f) in pheno_invalid: #skip that file
      continue
    sub_id = extract_id(f)
    # if sub_id == '50642 ':
    #   ipdb.set_trace()
    f_path = os.path.join(folder_path, f)
    data = np.loadtxt(f_path,dtype=np.float32)
    # ipdb.set_trace()
    df = pd.DataFrame(data)
    if df.isna().any().any():
      print("Missing values detected in fMRI time series!", f)
      missing_rois = df.isna().sum(axis=1)
      print("ROIs with missing values:", df.index[missing_rois].tolist())
      break
    # calc the pearson correlation-matrix for each .1D file
    # ipdb.set_trace()
    correlation_matrix = df.corr(method='pearson') #now, correlation_matrix is not a df but a np array
    if correlation_matrix.isna().any().any():
      print("NaN present in correlation matrix")
      # invalid_files.append(f)
      break
    # ipdb.set_trace()
    for i, j in extreme_pairs:
      subject_features.append(correlation_matrix.iloc[i, j])  # extract the same (i,j)
    features[sub_id] = subject_features

  # with open("/content/drive/MyDrive/project-2/diagnet_comp_final.pkl", "wb") as f:
  #     pickle.dump(features, f)
  return features


#the below function is used for generating our feature vector
def get_feature_vector_full(fc_path, invalid_files, pheno_invalid):
  # c = 0
  features = {}
  for f in os.listdir(fc_path):
    if extract_id(f) in invalid_files or extract_id(f) in pheno_invalid:
      continue
    f_path = os.path.join(fc_path, f)
    data = np.loadtxt(f_path,dtype=np.float32)
    df = pd.DataFrame(data)
    correlation_matrix = df.corr(method='pearson').values
    upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
    upper_tri_values = correlation_matrix[upper_tri_indices]  # shape will be (19900,)
    features[extract_id(f)] = upper_tri_values

    # c+=1
  
  return features
