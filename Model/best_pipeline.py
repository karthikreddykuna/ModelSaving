import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('ModelSaving/APPCode/prepared_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -48168.533361094946
exported_pipeline = DecisionTreeRegressor(max_depth=7, min_samples_leaf=10, min_samples_split=15)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
