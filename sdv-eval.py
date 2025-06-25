# import pickle

# path = "./data_census_onehot/transformed_columns.pk"
# with open(path, 'rb') as f:
#   encoder = pickle.load(f)

# path = "./data_census_onehot/missing_ratio-0.2_seed-1_max-min_norm.pk"
# with open(path, 'rb') as f:
#   contents = pickle.load(f)

# path = "./data_census_onehot/missing_ratio-0.2_seed-1.pk"
# with open(path, 'rb') as f:
#   content = pickle.load(f)


from sdv.evaluation import evaluate
try:
    from sdgym.datasets import load_dataset
except:
    pass
from sdgym.datasets import load_tables
from sdgym.datasets import load_dataset

import pandas as pd
import numpy as np

metadata = load_dataset('adult')
tables = load_tables(metadata)
data = tables['adult']

new_data = pd.read_csv('./generated_samples_census_0.2.csv')


#eval 1
# print(evaluate(new_data.iloc[:1000,:], data))

#eval 2
from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport

metadata = {
"fields": {
"age": {
"type": "numerical",
"subtype": "integer"
},
"workclass": {
"type": "categorical",
"pii": True
},
"fnlwgt": {
"type": "numerical",
"subtype": "integer"
},
"education": {
"type": "categorical"
},
"education-num": {
"type": "numerical",
"subtype": "integer"
},
"marital-status": {
"type": "categorical",
"pii": True
},
"occupation": {
"type": "categorical",
"pii": True
},
"relationship": {
"type": "categorical",
"pii": True
},
"race": {
"type": "categorical",
"pii": True
},
"sex": {
"type": "categorical",
"pii": True
},
"capital-gain": {
"type": "numerical",
"subtype": "integer"
},
"capital-loss": {
"type": "numerical",
"subtype": "integer"
},
"hours-per-week": {
"type": "numerical",
"subtype": "integer"
},
"native-country": {
"type": "categorical",
"pii": True
},
"label": {
"type": "categorical",
"pii": True
},
}
}

my_report = QualityReport()
my_report.generate(data, new_data, metadata)

print(my_report.get_details('Column Shapes'))

# my_report.get_visualization(property_name='Column Pair Trends')