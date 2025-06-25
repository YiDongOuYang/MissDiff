import numpy as np
import pandas as pd
import category_encoders as ce
import pickle
import ipdb

def reverse_transformation_bayesian(missing_rate=""):
    path = './data_bayesian_onehot/bayesian.csv'
    data = pd.read_csv(path, header=None)
    data.drop(0, axis=0, inplace=True)
    data.drop(columns=0, axis=1,inplace=True)
    data.columns = [i for i in range(5)]
    data.replace(" ?", np.nan, inplace=True)

    cat_list = [2, 3, 4]

    cont_list = [i for i in range(0, data.shape[1] - len(cat_list))]
    cat_list = [i for i in range(len(cont_list), data.shape[1])]

    encoder = ce.one_hot.OneHotEncoder(cols=data.columns[cat_list])
    encoder.fit(data)
    trans_data = encoder.transform(data)


    # new_observed_values = np.nan_to_num(new_observed_values)

    col_num = len(cont_list)
    data_cont = trans_data.to_numpy()[:,:col_num].astype(float)
    max_arr = np.nanmax(data_cont, axis=0)
    min_arr = np.nanmin(data_cont, axis=0)



    new_data=np.load('./generated_samples_bayesian3_'+missing_rate+'.npy')

    with open("./data_bayesian_onehot/transformed_columns.pk", "rb") as f:
        cont_cols, saved_cat_dict = pickle.load(f)

    index= [saved_cat_dict[key][0] for i, key in enumerate(saved_cat_dict)]

    index.append(new_data.shape[1])
    import torch
    new_data = torch.from_numpy(new_data).float() 
    for i in range(len(index)-1):
        new_data[:,index[i]:index[i+1]] = torch.nn.functional.softmax(new_data[:,index[i]:index[i+1]],1)
        aaa = torch.argmax(new_data[:,index[i]:index[i+1]], dim=1)
        bbb = torch.zeros(aaa.shape[0], aaa.max() + 1)
        bbb[torch.arange(aaa.shape[0]), aaa] = 1
        new_data[:,index[i]:index[i+1]] = bbb.int()


    excel_data = encoder.inverse_transform(new_data.numpy())
    excel_data = excel_data.to_numpy()
    excel_data[:,:col_num] = excel_data[:,:col_num] * (max_arr - min_arr) + min_arr

    excel_data = pd.DataFrame(excel_data)



    excel_data.to_csv('./generated_samples_bayesian3_'+missing_rate+'.csv', index=False, header=['temperature','humidity','person','ac_on','ac_off'])


