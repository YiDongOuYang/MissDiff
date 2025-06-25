#!/bin/bash 

for ((i=1; i<10; i++))  
do  
python exe_bayesian_onehot.py --testmissingratio=i/10
python reverse_transformation_bayesian.py
python calculate_likelihood_bayesian.py
done  






