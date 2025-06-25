import argparse
import torch
import datetime
import json
import yaml
import os

from src.main_model_table import CSDIT
from src.utils_table import train, evaluate_onehot,generation

from dataset_bayesian_onehot import get_dataloader

import numpy as np
import ipdb

parser = argparse.ArgumentParser(description="CSDI_T")
parser.add_argument("--config", type=str, default="census_onehot_analog.yaml")
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
# parser.add_argument("--testmissingratio", type=float, default=0.8)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

log_likelihood = []
number=0
for testmissingratio in np.arange(0.1,1,0.1):

    args.modelfolder = ['bayesian5_20230422_183028','bayesian5_20230422_184132','bayesian5_20230422_185228','bayesian5_20230422_190320','bayesian5_20230422_191416','bayesian5_20230422_192511','bayesian5_20230422_193602','bayesian5_20230422_194648','bayesian5_20230422_195732'][number]
    number+=1

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["test_missing_ratio"] = testmissingratio#args.testmissingratio

    print(json.dumps(config, indent=4))

    # Create folder
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/bayesian" + str(args.nfold) + "_" + current_time + "/"
    print("model folder:", foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
    train_loader, valid_loader, test_loader = get_dataloader(
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
    )
    exe_name = "bayesian"
    model = CSDIT(config, args.device).to(args.device)

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    print("---------------Start testing---------------")
    # evaluate_onehot(
    #     exe_name, model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername
    # )
    # ipdb.set_trace()
    log_likelihood.append(generation(model,config["model"]["test_missing_ratio"]))

# import pandas as pd
# dict = dict = {"missing_ratio":np.arange(0.1,1,0.1),"log_likelihood":log_likelihood }
# df = pd.DataFrame(dict) 
# df.to_csv('miss_diff_bayesian_row_missing2.csv')