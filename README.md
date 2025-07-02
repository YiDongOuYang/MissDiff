# MissDiff: Training Diffusion Models on Tabular Data with Missing Values

This repo contains the pytorch code for experiments in the paper [MissDiff: Training Diffusion Models on Tabular Data with Missing Values](https://arxiv.org/abs/2307.00467)

by Yidong Ouyang, Liyan Xie, Chongxuan Li, Guang Cheng.

We present a unified and principled diffusion-based framework for learning from data with missing values and generating synthetic complete data. Our method models the score of complete data distribution by denoising score matching on data with missing values. We prove that the proposed method can recover the score of the complete data distribution, and the proposed training objective serves as an upper bound for the negative likelihood of observed data.

### Usage 

Training MissDiff for Bayesian Dataset
```
python exe_bayesian_onehot.py
```
Convert the synthetic data from npy to csv file
```
python reverse_transformation_bayesian.py
```
Utility evaluation

```
python utility.py

```

Fidelity evaluation

```
python evaluation.py
```

### References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{
  Ouyang2023miss,
  title={MissDiff: Training Diffusion Models on Tabular Data with Missing Values},
  author={Yidong Ouyang, Liyan Xie, Chongxuan Li, Guang Cheng},
  booktitle={ICML workshop on Structured Probabilistic Inference \& Generative Modeling},
  year={2023},
}
```

This implementation is heavily based on 
* [CSDI_T](https://github.com/pfnet-research/TabCSDI) 
* [hyperimpute](https://github.com/vanderschaarlab/hyperimpute) 
