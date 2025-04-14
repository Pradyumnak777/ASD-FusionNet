import pandas as pd
import numpy as np


def extract_id(s):
    match = re.search(r'0*([1-9]\d{4})', s)
    return match.group(1) if match else None
  

def get_label_dict(dict): #any dict -> features/pheno can be passed as long as both have been filtered
    label_dict = {}
    csv_path = '/content/s3bucket/data/Projects/ABIDE/Phenotypic_V1_0b_preprocessed.csv'
    df = pd.read_csv(csv_path)
    for k,v in dict.items():
        # ipdb.set_trace()
        label = df.loc[df['SUB_ID'] == int(k), 'DX_GROUP'].values[0]
        label_dict[k] = int(label - 1)

    return label_dict
