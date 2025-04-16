from utils.data_processing import new_features, new_pheno_dict, label_dict
from utils.training import cross_validate_fusion
import torch
import pickle



features_path = '/path/to/features.pkl'
pheno_dict_path = '/path/to/pheno_dict.pkl'
label_dict_path = '/path/to/label_dict.pkl'

with open(features_path, 'rb') as f:
    new_features = pickle.load(f)

with open(pheno_dict_path, 'rb') as f:
    new_pheno_dict = pickle.load(f)

with open(label_dict_path, 'rb') as f:
    label_dict = pickle.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cross_validate_fusion(
    conn_dict=new_features,
    pheno_dict=new_pheno_dict,
    label_dict=label_dict,
    device=device,
    n_splits=5,
    epochs=100,
    eval_only= True
)

