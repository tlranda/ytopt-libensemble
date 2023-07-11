#!/usr/bin/env python
# coding: utf-8

import shap
import nest_asyncio
nest_asyncio.apply()
#!export TF_CPP_MIN_LOG_LEVEL=3
# Avoid logging TF's DEBUG/INFO/WARNING statements
#!export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

IN_FILE = 'results256-256-mpi-4.csv'
real_data = pd.read_csv(IN_FILE)
# Fix the data so it detects numerical / categorical correctly
for idx in range(7):
    real_data[f'p{idx}'] = real_data[f'p{idx}'].apply(lambda x: str(x[2:-2]) if x[1] == "'" else int(x[1:-1]))

#real_data.describe
#print(real_data)

objective = 'RUNTIME'
real_data['objective'] = -1 * real_data[objective]
#real_df = df.loc[df['objective'] < q_10]
real_df = real_data.drop(columns=['RUNTIME','elapsed_sec'])
print("Data:")
print(real_df)

print('non-unique columns:')
print([e for e in real_df.columns if real_df[e].nunique() != 1])

fig1, ax1 = plt.subplots()
ax1.hist(real_df.objective.values)
ax1.set_xlabel('Objective Values')
ax1.set_ylabel('# Occurrences')
ax1.set_title(f'Histogram of objectives for {IN_FILE}')

real_df_cut = real_df#[real_df.objective < 1000.00]
#real_df_cut

# determine categorical and numerical features
numerical_ix = real_df_cut.select_dtypes(include=['int64', 'float64']).columns
categorical_ix = real_df_cut.select_dtypes(include=['object', 'bool']).columns

print("numerical features:")
print(numerical_ix)
print("categorical_features:")
print(categorical_ix)

from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, input_array, y=None):
        return self
    def transform(self, input_array, y=None):
        return input_array*1

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder, MinMaxScaler, OrdinalEncoder
import skopt
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor

t = [('cat', OrdinalEncoder(), categorical_ix), ('num', IdentityTransformer(), numerical_ix)]
data_pipeline = ColumnTransformer(transformers=t)

data_pipeline_model = data_pipeline.fit(real_df_cut)

preprocessed_data = data_pipeline_model.transform(real_df_cut)

print("Shape of preprocessed data:")
print(preprocessed_data.shape)

fig2, ax2 = plt.subplots()
ax2.scatter(preprocessed_data[:,7],real_df_cut['objective'].values)
ax2.set_ylabel('actual objective values')
ax2.set_xlabel('preprocessed objectives')
ax2.set_title(f"Preprocessed vs Actual objecties for {IN_FILE}")
#preprocessed_data[:,7]

regr = RandomForestRegressor()
#regr.fit(preprocessed_data[:,1:],preprocessed_data[:,0])

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(regr, preprocessed_data[:,0:7], preprocessed_data[:,7], cv=10)

print("Initial CV scores:")
print(cv_scores)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data[:,0:7], preprocessed_data[:,7], test_size=0.50,
                                                    random_state=42)
print("X_train shape:")
print(X_train.shape)

regr.fit(X_train,y_train)

preds = regr.predict(preprocessed_data[:,0:7])
fig3, ax3 = plt.subplots()
ax3.scatter(preprocessed_data[:,7],preds)
r2 = r2_score(preprocessed_data[:,7],preds)
print("Regression's R2 score:")
print(r2)
ax3.set_ylabel('Predicted Wrench error')
ax3.set_xlabel('Observed Wrench error')
#ax3.suptitle('Wrench simulation error prediction')
ax3.set_title('R2=%1.3f'%r2)
ax3.grid()
plt.show()

input_names = real_df_cut.columns.to_list()[:-1]

sorted_idx = regr.feature_importances_.argsort()

print("Features sorted by importance:")
print(sorted_idx)
print(np.asarray(input_names)[sorted_idx])

input_names_new = []
for inp in input_names:
    k = inp
    input_names_new.append(k)

input_names = input_names_new

import shap

explainer = shap.TreeExplainer(regr)
shap_values = explainer.shap_values(preprocessed_data[:,0:7])
explanation = shap.Explanation(shap_values, feature_names=input_names)
shap.plots.beeswarm(explanation)
shap.summary_plot(shap_values, feature_names = input_names, plot_type="bar")

X_test1 = pd.DataFrame(preprocessed_data[:,0:7])
X_test1.columns = input_names

#print(X_test1)

random_indices = random.sample(range(X_test1.shape[0]),1)

for i in random_indices:
    choosen_instance = X_test1.loc[[i]]
    shap_values = explainer.shap_values(choosen_instance)
    #shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values, choosen_instance, matplotlib=True)

