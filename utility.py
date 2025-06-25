from evaluation import compute_scores
import pandas as pd
import numpy as np

import ipdb

all_mean_f1 = []
all_std_f1 = []

for missing_ratio in np.arange(0.1,1,0.1):
    path = './generated_samples_bayesian2_'+str(round(missing_ratio,1))+'.csv'
    synthesized_data = pd.read_csv(path, header=None)
    synthesized_data.drop(0, axis=0, inplace=True)
    # synthesized_data[0].astype(float)
    # ipdb.set_trace()

    path = './data_bayesian_onehot/bayesian.csv'
    train = pd.read_csv(path, header=None)
    train.drop(0, axis=0, inplace=True)
    train.drop(columns=0, axis=1,inplace=True)
    train.columns = [i for i in range(5)]

    path = './data_bayesian_onehot/bayesian_test.csv'
    test = pd.read_csv(path, header=None)
    test.drop(0, axis=0, inplace=True)
    test.drop(columns=0, axis=1,inplace=True)
    test.columns = [i for i in range(5)]

    metadata = {'problem_type' : 'multiclass_classification','columns':[{
                "max": 31.67, 
                "min": 18.75, 
                "name": 0, 
                "type": "continuous"
            }, 
            {
                "max": 70.58, 
                "min": 33.66, 
                "name": 1, 
                "type": "continuous"
            }, 
            {
                "i2s": [
                            "1", 
                            "2", 
                        ], 
                "size": 2, 
                "name": 2, 
                "type": "categorical"
            }, 
            {
                "i2s": [
                            "1", 
                            "2", 
                            "3", 
                        ], 
                "size": 3,  
                "name": 3, 
                "type": "categorical"
            }, 
            {
                "i2s": [
                            "1", 
                            "2", 
                        ], 
                "size": 2, 
                "name": 4, 
                "type": "categorical"
            }, ]}

    mean_f1, std_f1 = compute_scores(train.to_numpy(), test.to_numpy(), synthesized_data.to_numpy(), metadata)
    all_mean_f1.append(mean_f1)
    all_std_f1.append(std_f1)

dict = dict = {"missing_ratio":np.arange(0.1,1,0.1),"f1_mean":all_mean_f1, "f1_std":all_std_f1}
df = pd.DataFrame(dict) 
df.to_csv('miss_diff_bayesian_utility2.csv')
print(all_mean_f1)
print(all_std_f1)