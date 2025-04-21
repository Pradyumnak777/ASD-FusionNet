from ML import cross_validate_fusion, FusionDataset
from fc import fc_invalid, get_feature_vector_full
from pheno import get_pheno_invalid, get_pheno_dict
from aux_functions import get_label_dict
import torch
import pickle

#get the invalid files from functional connectivity features
fc_path = '/mnt/c/Users/Lenovo/OneDrive/ASD-BranchNet/s3bucket_shortcut/data/Projects/ABIDE/Outputs/cpac/filt_noglobal'
fc_invalid_lst = fc_invalid(fc_path)

#get invalid files from phenotypic feature group - [VIQ, PIQ, func_mean_fd, func_perc_fd, func_num_fd, func_quality]
pheno_invalid, initial_pheno_dict = get_pheno_invalid() # the file path is stored within the function itself, no need to parse

#now, extract 19900 features per subject frmo the original 40000 by considering only the upper triangle of the matrix
fc_feature_dict = get_feature_vector_full(fc_path, fc_invalid, pheno_invalid)

#get the phenotypic features
pheno_feature_dict = get_pheno_dict(initial_pheno_dict, pheno_invalid, fc_invalid)

#now generate the label dict, any of the above 2 dictionaries can be passed(as we have already used common keys)
label_dict = get_label_dict(pheno_feature_dict)



#load readymade features- (for demonstration this can be used) NAMES AND GDRIVE LOCATION WILL VARY, DEPENDING ON WHERE YOU SET UP

# features_path = 'gdrive_link/final_feature_dict_proj2.pkl'
# pheno_dict_path = 'gdrive_link/final_pheno_dict.pkl'
# label_dict_path = 'gdrive_link/final_label_dict.pkl'

# with open(features_path, 'rb') as f:
#     fc_feature_dict = pickle.load(f)

# with open(pheno_dict_path, 'rb') as f:
#     pheno_feature_dict = pickle.load(f)

# with open(label_dict_path, 'rb') as f:
#     label_dict = pickle.load(f)


#now, pass this to the cross-validate function

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cross_validate_fusion(
    conn_dict=fc_feature_dict,
    pheno_dict=pheno_feature_dict,
    label_dict=label_dict,
    device=device,
    n_splits=5,
    epochs=100,
    eval_only= True
)

