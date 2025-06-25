import pickle
import yaml
import os
import math
import re
import numpy as np
import pandas as pd
import category_encoders as ce

from torch.utils.data import DataLoader, Dataset

import ipdb

def process_func(path: str, cat_list, missing_ratio=0.1, encode=True):
    data = pd.read_csv(path, header=None)
    data.drop(0, axis=0, inplace=True)
    data.drop(columns=0, axis=1,inplace=True)
    data.columns = [i for i in range(5)]
    data.replace(" ?", np.nan, inplace=True)

    # Swap columns
    # temp_list = [i for i in range(data.shape[1]) if i not in cat_list]
    # temp_list.extend(cat_list)
    # new_cols_order = temp_list
    # data = data.reindex(columns=data.columns[new_cols_order])
    # data.columns = [i for i in range(data.shape[1])]

    # create two lists to store position
    cat_list = [2, 3, 4]
    cont_list = [i for i in range(0, data.shape[1] - len(cat_list))]
    cat_list = [i for i in range(len(cont_list), data.shape[1])]

    observed_values = data.values
    observed_masks = ~pd.isnull(data)
    observed_masks = observed_masks.values

    masks = observed_masks.copy()

    '''
    original code
    # In this section, obtain gt_masks
    # for each column, mask {missing_ratio} % of observed values.
    for col in range(masks.shape[1]):  # col #
        obs_indices = np.where(masks[:, col])[0]
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices, col] = False
    '''
    for missing_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # for each row, mask {missing_ratio} % of observed values. Using for learning from missing data
        observed_masks = np.random.choice([0, 1],size=observed_values.shape, p=[missing_ratio, 1-missing_ratio])
        all_ones_mask = np.all(observed_masks == 1, axis=1)
        count_all_ones = np.sum(all_ones_mask)
        print(count_all_ones)
    ipdb.set_trace()

    masks = observed_masks.copy()

    observed_values = observed_values * observed_masks  #using 0 for imputation
    
    '''
    # using mean for imputation

    #even for mean imputation, we still need 
    observed_values = observed_values * observed_masks

    observed_values[observed_values == 0] = np.nan
    mean_for_imputation = np.nanmean(observed_values, axis=0)
    missing_index = np.argwhere(np.isnan(observed_values))
    for _, (x,y) in enumerate(missing_index):
        observed_values[x,y] = mean_for_imputation[y]
    '''

    # for col in range(observed_values.shape[0]):  # row #
    #     obs_indices = np.where(masks[col,: ])[0]
    #     miss_indices = np.random.choice(
    #         obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    #     )
    #     masks[col,miss_indices ] = False

    # gt_mask: 0 for missing elements and manully maksed elements
    gt_masks = masks.reshape(observed_masks.shape)
    
    if encode == True:
        # set encoder here
        encoder = ce.one_hot.OneHotEncoder(cols=data.columns[cat_list])
        encoder.fit(data)
        new_df = encoder.transform(data)

        # we now need to transform these masks to the new one, suitable for mixed data types.
        cum_num_bits = 0
        new_observed_masks = observed_masks.copy()
        new_gt_masks = gt_masks.copy()

        for index, col in enumerate(cat_list):
            corresponding_cols = len(
                [
                    s
                    for s in new_df.columns
                    if isinstance(s, str) and s.startswith(str(col) + "_")
                ]
            )
            add_col_num = corresponding_cols
            insert_col_obs = observed_masks[:, col]
            insert_col_gt = gt_masks[:, col]

            for i in range(add_col_num - 1):
                new_observed_masks = np.insert(
                    new_observed_masks, cum_num_bits + col, insert_col_obs, axis=1
                )
                new_gt_masks = np.insert(
                    new_gt_masks, cum_num_bits + col, insert_col_gt, axis=1
                )
            cum_num_bits += add_col_num - 1

        new_observed_values = new_df.values
        new_observed_values_cat = new_observed_values[:, len(cont_list) :]
        index = new_observed_values_cat == 0
        new_observed_values_cat[index] = -1
        new_observed_values[:, len(cont_list) :] = new_observed_values_cat
        new_observed_values = np.nan_to_num(new_observed_values)
        new_observed_values = new_observed_values.astype(np.float)

        saved_cat_dict = {}
        for index, col in enumerate(cat_list):
            indices = [
                i
                for i, s in enumerate(new_df.columns)
                if isinstance(s, str) and s.startswith(str(col))
            ]
            saved_cat_dict[str(cat_list[index])] = indices

        with open("./data_bayesian_onehot/transformed_columns.pk", "wb") as f:
            pickle.dump([cont_list, saved_cat_dict], f)

        with open("./data_bayesian_onehot/encoder.pk", "wb") as f:
            pickle.dump(encoder, f)

    if encode == True:
        new_observed_values = new_observed_values * new_observed_masks
        return new_observed_values, new_observed_masks, new_gt_masks, cont_list
    else:
        cont_cols = [i for i in data.columns if i not in cat_list]
        return observed_values, observed_masks, gt_masks, cont_list


class tabular_dataset(Dataset):
    def __init__(self, eval_length=9, use_index_list=None, missing_ratio=0.1, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)

        dataset_path = "./data_bayesian_onehot/bayesian.csv"
        processed_data_path = (
            f"./data_bayesian_onehot/missing_ratio-{missing_ratio}_seed-{seed}.pk"
        )
        processed_data_path_norm = f"./data_bayesian_onehot/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"

        # self.cont_cols is only saved in .pk file before normalization.
        cat_list = [2, 3, 4]                  
        if not os.path.isfile(processed_data_path):
            (
                self.observed_values,
                self.observed_masks,
                self.gt_masks,
                self.cont_cols,
            ) = process_func(
                dataset_path,
                cat_list=cat_list,
                missing_ratio=missing_ratio,
                encode=True,
            )

            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [
                        self.observed_values,
                        self.observed_masks,
                        self.gt_masks,
                        self.cont_cols,
                    ],
                    f,
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):  # load datasetfile
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
            print("--------Normalized dataset loaded--------")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1):
    dataset = tabular_dataset(missing_ratio=missing_ratio, seed=seed)
    print(f"Dataset size:{len(dataset)} entries")

    indlist = np.arange(len(dataset))

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)
   
    remain_index = indlist

    np.random.shuffle(remain_index)
    num_train = (int)(len(remain_index) * 1)
    train_index = remain_index[:num_train]

    # Here we perform max-min normalization.
    processed_data_path_norm = f"./data_bayesian_onehot/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
    if not os.path.isfile(processed_data_path_norm):
        print(
            "--------------Dataset has not been normalized yet. Perform data normalization and store the mean value of each column.--------------"
        )
        # data transformation after train-test split.
        col_num = len(dataset.cont_cols)
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        for index, k in enumerate(dataset.cont_cols):
            # Using observed_mask to avoid including missing values (now represented as 0)
            obs_ind = dataset.observed_masks[train_index, k].astype(bool)
            temp = dataset.observed_values[train_index, k]
            max_arr[index] = max(temp[obs_ind])
            min_arr[index] = min(temp[obs_ind])

        print(
            f"--------------Max-value for cont-variable column {max_arr}--------------"
        )
        print(
            f"--------------Min-value for cont-variable column {min_arr}--------------"
        )

        for index, k in enumerate(dataset.cont_cols):
            dataset.observed_values[:, k] = (
                (dataset.observed_values[:, k] - (min_arr[index] - 1))
                / (max_arr[index] - min_arr[index] + 1)
            ) * dataset.observed_masks[:, k]

        with open(processed_data_path_norm, "wb") as f:
            pickle.dump(
                [dataset.observed_values, dataset.observed_masks, dataset.gt_masks], f
            )

    # Create datasets and corresponding data loaders objects.
    train_dataset = tabular_dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_loader = None

    test_loader = None

    print(f"Training dataset size: {len(train_dataset)}")

    return train_loader, valid_loader, test_loader
