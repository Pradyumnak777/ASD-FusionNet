import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


# def get_pheno_invalid(pheno_dict):
#     # c = 0
    
#     return pheno_invalid

# df['DX_GROUP'] = df['DX_GROUP'].map({1: 1, 2: 0})  # 1 = ASD, 0 = Control
def get_pheno_invalid():
    df = pd.read_csv('/content/s3bucket/data/Projects/ABIDE/Phenotypic_V1_0b_preprocessed.csv')
    pheno_dict = {}
    for sub_id in df['SUB_ID']:
        func_mean_fd = df.loc[df['SUB_ID'] == sub_id, 'func_mean_fd'].values[0]
        func_perc_fd = df.loc[df['SUB_ID'] == sub_id, 'func_perc_fd'].values[0]
        func_num_fd = df.loc[df['SUB_ID'] == sub_id, 'func_num_fd'].values[0]
        func_quality = df.loc[df['SUB_ID'] == sub_id, 'func_quality'].values[0]
        viq = df.loc[df['SUB_ID'] == sub_id, 'VIQ'].values[0]
        piq = df.loc[df['SUB_ID'] == sub_id, 'PIQ'].values[0]

        pheno_dict[str(sub_id)] = [func_mean_fd, func_perc_fd, func_num_fd, func_quality, viq, piq]
        #   pheno_dict[str(sub_id)] = [func_mean_fd, func_perc_fd, func_num_fd, viq]
    
    pheno_invalid = []    
    for key,val in pheno_dict.items():
        arr = np.array(val)
        if np.isnan(arr).any() or (-9999 in arr):
            pheno_invalid.append(key)
            
            # print(f"{key} Contains NaN")
            # c+=1
    return pheno_invalid, pheno_dict


def get_pheno_dict(initial_pheno_dict, pheno_invalid, invalid_files):
    sub_ids = list(initial_pheno_dict.keys())
    pheno_values = np.array([initial_pheno_dict[k] for k in sub_ids])  # shape: (n_subjects, 2)

    # Step 2: Normalize using StandardScaler (mean=0, std=1)
    scaler = StandardScaler()
    normalized_pheno = scaler.fit_transform(pheno_values)

    # Step 3: Reassign the normalized values back to the dictionary
    normalized_pheno_dict = {sub_ids[i]: normalized_pheno[i].tolist() for i in range(len(sub_ids))}
    
    new_pheno_dict = {}

    for k in normalized_pheno_dict.keys():
        if k in pheno_invalid or k in invalid_files:
        # if k in pheno_invalid:
            continue
        new_pheno_dict[k] = normalized_pheno_dict[k]

    return new_pheno_dict
        

    