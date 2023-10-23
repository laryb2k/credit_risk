import pandas as pd
import numpy as np
import os
import datetime
import random
import sys
import time

import xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import shap
import truera
from truera_qii import qii
from truera_qii.qii.explainers import tree, linear
from truera.client.ingestion import add_data, ColumnSpec, ModelOutputContext
from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import BasicAuthentication, TokenAuthentication
start_time= datetime.datetime.now().replace(microsecond=0)

print("Reading in data...")
datasplit_dirname = 'datasplits'
dc_dirname = 'dc_v1'
split_names = [i for i in os.listdir(os.path.join(datasplit_dirname, dc_dirname)) if i.startswith('201') and len(i)==6]
data = {}
for i in split_names:
    data[i] = {
        'data': pd.read_csv(os.path.join(datasplit_dirname, dc_dirname, i, f'data_{i}.csv')),
        'data_raw': pd.read_csv(os.path.join(datasplit_dirname, dc_dirname, i, f'data_raw_{i}.csv')),
        'label': pd.read_csv(os.path.join(datasplit_dirname, dc_dirname, i, f'label_{i}.csv'), header=None)
    }

    
#Define train data    
train_post = pd.concat([data.get('2018Q1').get('data'), data.get('2018Q2').get('data'), 
                        data.get('2018Q3').get('data'), data.get('2018Q4').get('data')]).reset_index(drop=True)

train_pre = pd.concat([data.get('2018Q1').get('data_raw'), data.get('2018Q2').get('data_raw'), 
                       data.get('2018Q3').get('data_raw'), data.get('2018Q4').get('data_raw')]).reset_index(drop=True)

train_labels =  pd.concat([data.get('2018Q1').get('label'), data.get('2018Q2').get('label'), 
                       data.get('2018Q3').get('label'), data.get('2018Q4').get('label')]).reset_index(drop=True)

#Define prod data
prod_post = pd.concat([data.get('2016Q3').get('data'), data.get('2017Q4').get('data'),
                        data.get('2017Q1').get('data'), data.get('2017Q2').get('data'), 
                        data.get('2017Q3').get('data'), data.get('2017Q4').get('data')]).reset_index(drop=True)

prod_pre = pd.concat([data.get('2016Q3').get('data_raw'), data.get('2016Q4').get('data_raw'),
                       data.get('2017Q1').get('data_raw'), data.get('2017Q2').get('data_raw'), 
                       data.get('2017Q3').get('data_raw'), data.get('2017Q4').get('data_raw')]).reset_index(drop=True)

prod_labels =  pd.concat([data.get('2016Q3').get('label'), data.get('2016Q4').get('label'),
                       data.get('2017Q1').get('label'), data.get('2017Q2').get('label'), 
                       data.get('2017Q3').get('label'), data.get('2017Q4').get('label')]).reset_index(drop=True)

#Cast bool vars to float
train_pre['debt_settlement_flag']=train_pre['debt_settlement_flag'].astype(float)
prod_pre['debt_settlement_flag']=prod_pre['debt_settlement_flag'].astype(float)

#Add labels to pre data dfs
prod_pre['label'] = prod_labels
train_pre['label'] = train_labels

#Resample prod data
prod_pre = prod_pre[prod_pre.label==1].append(prod_pre[prod_pre.label==0].sample(n=32014, random_state=1))
prod_pre = prod_pre.sample(n=len(prod_pre), random_state=2)

#Filter prod post data to match pre data and reset indexes
prod_post = prod_post.filter(items=prod_pre.index, axis=0)
prod_pre.reset_index(inplace=True, drop=True)
prod_post.reset_index(inplace=True, drop=True)

#Define list of pre data columns
pre_transform_cols = prod_pre.drop("label", axis=1).columns

#Define feature map
FEATURE_MAP = {}
for post in train_post.columns:
    mapped = None
    for pre in train_pre.columns:
        if post.startswith(pre) and (mapped is None or len(mapped) < len(pre)):
            mapped = pre
    if mapped not in FEATURE_MAP:
        FEATURE_MAP[mapped] = []
    FEATURE_MAP[mapped].append(train_post.columns.get_loc(post))
#Define categorical feature list    
catFeats = list(train_pre.columns[train_pre.dtypes == 'O'])

#Build XGB pipeline
print("Training XGBoost Model")
scale_weight = train_labels.value_counts()[0]/train_labels.value_counts()[1]
xgb = XGBClassifier(booster = "gbtree", n_estimators=25, max_depth=4, scale_pos_weight = scale_weight, random_state=1)
xgb.fit(train_post, train_pre.label)

#Get predictions
xgb_train_preds = xgb.predict(train_post)
xgb_prod_preds = xgb.predict(prod_post)

print("XGB Train Performance:")
print(classification_report(train_pre.label, xgb_train_preds))

print("XGB Prod Performance:")
print(classification_report(prod_pre.label, xgb_prod_preds))


print("Training Random Forest Model")
rf = RandomForestClassifier(n_estimators = 10, random_state=1)
rf.fit(train_post, train_pre.label)
#Get predictions
rf_train_preds = rf.predict(train_post)
rf_prod_preds = rf.predict(prod_post)

print("RF Train Performance:")
print(classification_report(train_pre.label, rf_train_preds))

print("RF Prod Performance:")
print(classification_report(prod_pre.label, rf_prod_preds))

from sklearn.metrics import roc_auc_score
print("xgb train AUC", roc_auc_score(train_pre.label, xgb_train_preds))
print("xgb prod AUC", roc_auc_score(prod_pre.label, xgb_prod_preds))
print("rf train AUC",roc_auc_score(train_pre.label, rf_train_preds))
print("rf prod AUC",roc_auc_score(prod_pre.label, rf_prod_preds))

#Compute feature influences if not already computed for train/prod for each model
try: 
    xgb_fi_train = pd.read_csv("FI_probits/xgb_train_FI.csv")
    print("Found existing FI data for xgb train")
except:
    print("Computing FIs and writing locally (will take a while)")
    xgb_explainer = qii.explainers.tree.TreeExplainer(xgb, shap.sample(train_post,100),  
                                                      pretransform_features=pre_transform_cols, feature_map=FEATURE_MAP, model_output="probability")
    xgb_fi_train = pd.DataFrame(xgb_explainer.truera_qii_values(train_post, train_pre.label), columns = pre_transform_cols)
    xgb_fi_train.to_csv("FI_probits/xgb_train_FI.csv",index=False) 
    
try: 
    xgb_fi_prod = pd.read_csv("FI_probits/xgb_prod_FI.csv")
    print("Found existing FI data for xgb prod")
except:
    print("Computing FIs and writing locally (will take a while)")
    xgb_explainer = qii.explainers.tree.TreeExplainer(xgb, shap.sample(train_post,100),  
                                                  pretransform_features=pre_transform_cols, feature_map=FEATURE_MAP, model_output="probability")
    xgb_fi_prod = pd.DataFrame(xgb_explainer.truera_qii_values(prod_post, prod_pre.label), columns = prod_pre.columns)
    xgb_fi_prod.to_csv("FI_probits/xgb_prod_FI.csv",index=False)
    
try: 
    rf_fi_train = pd.read_csv("FI_probits/rf_train_FI.csv")
    print("Found existing FI data for rf train")
except:
    print("Computing FIs and writing locally (will take a while)")
    rf_explainer = qii.explainers.tree.TreeExplainer(rf, shap.sample(train_post, 100), 
                                                 pretransform_features=pre_transform_cols, feature_map=FEATURE_MAP, model_output="probability")
    rf_fi_train = pd.DataFrame(rf_explainer.truera_qii_values(train_post, label), columns = train_pre.columns)
    rf_fi_train.to_csv("FI_probits/rf_train_FI.csv",index=False)
    

try: 
    rf_fi_prod = pd.read_csv("FI_probits/rf_prod_FI.csv")
    print("Found existing FI data for rf prod")
except:
    print("Computing FIs and writing locally (will take a while)")
    rf_explainer = qii.explainers.tree.TreeExplainer(rf, shap.sample(train_post, 100), 
                                                     pretransform_features=pre_transform_cols, feature_map=FEATURE_MAP, model_output="probability")

    rf_fi_prod = pd.DataFrame(rf_explainer.truera_qii_values(prod_post, prod_pre.label), columns = pre_transform_cols)
    rf_fi_prod.to_csv("FI_probits/rf_prod_FI.csv",index=False)
    
    
#Compute prediction probabilities for models
rf_train_preds = rf.predict_proba(train_post)[:,1]
xgb_train_preds = xgb.predict_proba(train_post)[:,1]

rf_prod_preds = rf.predict_proba(prod_post)[:,1]
xgb_prod_preds = xgb.predict_proba(prod_post)[:,1]

#Rename FI columns
rf_fi_train.rename(columns={c: f"{c}_truera-qii_influence" for c in rf_fi_train.columns}, inplace=True)
xgb_fi_train.rename(columns={c: f"{c}_truera-qii_influence" for c in xgb_fi_train.columns}, inplace=True)

rf_fi_prod.rename(columns={c: f"{c}_truera-qii_influence" for c in rf_fi_prod.columns}, inplace=True)
xgb_fi_prod.rename(columns={c: f"{c}_truera-qii_influence" for c in xgb_fi_prod.columns}, inplace=True)

#Set up train and prod timestamps
train_pre['timestamp'] = pd.Series(train_pre.index).apply(lambda x: datetime.datetime(2023, 1, 1)+datetime.timedelta(minutes=x*2.25))
end_train = train_pre.timestamp.max()+datetime.timedelta(minutes=4.5)
prod_pre['timestamp'] = pd.Series(prod_pre.index).apply(lambda x: end_train+datetime.timedelta(minutes=x*2.5))

#Seperate data frames for each model
rf_train = train_pre.copy()
xgb_train = train_pre.copy()

rf_prod = prod_pre.copy()
xgb_prod= prod_pre.copy()

rf_train['id'] = pd.Series(train_pre.index).apply(lambda x: "rf_train_"+str(x))
xgb_train['id'] = pd.Series(train_pre.index).apply(lambda x: "xgb_train_"+str(x))

rf_prod['id'] = pd.Series(prod_pre.index).apply(lambda x: "rf_prod_"+str(x))
xgb_prod['id'] = pd.Series(prod_pre.index).apply(lambda x: "xgb_prod_"+str(x))

#Add IDs to feature influence data
rf_fi_train['id'] = rf_train.id
rf_fi_prod['id'] = rf_prod.id

xgb_fi_train['id'] = xgb_train.id
xgb_fi_prod['id'] = xgb_prod.id

#Add predictions to dfs
rf_train['prediction'] = np.hstack([rf_train_preds])
xgb_train['prediction'] = np.hstack([xgb_train_preds])

rf_prod['prediction'] = np.hstack([rf_prod_preds])
xgb_prod['prediction'] = np.hstack([xgb_prod_preds])

#Add labels to train (already exist in prod)
rf_train['label'] = train_labels
xgb_train['label'] = train_labels

print("Ensure all train dfs have same shape")
print(xgb_train.drop(["label","prediction", "timestamp"], axis=1).shape, rf_train.drop(["label","prediction", "timestamp"], axis=1).shape, rf_fi_train.shape, xgb_fi_train.shape)

print("Ensure all prod dfs have same shape")
print(xgb_prod.drop(["label","prediction", "timestamp"], axis=1).shape, rf_prod.drop(["label","prediction", "timestamp"], axis=1).shape, rf_fi_prod.shape, xgb_fi_prod.shape)

print("ensure id dtypes are the same")
print(rf_train.id.dtypes, rf_prod.id.dtypes)

print("RF dtype check pass: ", sum(rf_train.dtypes == rf_prod.dtypes)==rf_train.shape[1])
print("RF dtype check pass: ", sum(xgb_train.dtypes == xgb_prod.dtypes)==xgb_train.shape[1])

#Define and print key columns
prediction_col = "prediction"
label_col = "label"
timestamp_col = "timestamp"
id_col = "id"
pre_cols = list(rf_train.drop(["id","label","prediction", "timestamp"], axis=1).columns)
fi_cols = list(rf_fi_train.drop(["id"], axis=1).columns)

print("Prediction Column:", prediction_col)
print("Label Column:", label_col)
print("Timestamp Column:", timestamp_col)
print("ID Column:", id_col)


#Start Truera Ingestion
ingestion_start = datetime.datetime.now().replace(microsecond=0)

print("Make Sure to add your token from the UI. Docs on how to generate a token can be seen here https://docs.truera.com/1.40/public/diagnostics-quickstart/#connecting-to-truera")

token = "<INSERT YOUR TOKEN HERE>"

CONNECTION_STRING = "https://app.truera.net/"


### CHECK YOUR TOKEN/CONNECTION STRING!!!!

auth = TokenAuthentication(token)
tru = TrueraWorkspace(CONNECTION_STRING, auth, verify_cert=False)

#Specify to use truera server for remote compute
tru.set_environment("remote")

PROJECT_NAME = sys.argv[1]
print(PROJECT_NAME)

# Add new project
try: 
    tru.add_project(PROJECT_NAME, score_type="probits")
    print("Adding new project")
except:
    print("Setting existing project")
    tru.set_project(PROJECT_NAME)
    
#Add rf data collection
DATA_COLLECTION_NAME="Credit-risk-rf"

try:
    tru.add_data_collection(DATA_COLLECTION_NAME)
    print('Adding new data colleciton')
except:
    print('Setting existing data collection')
    tru.set_data_collection(DATA_COLLECTION_NAME)
    
#Upload random forest model
tru.add_model('Random_Forest')

rf_train_start = datetime.datetime.now().replace(microsecond=0)

#Define column_spec
column_spec = ColumnSpec(
        id_col_name=id_col,
        timestamp_col_name = timestamp_col,
        prediction_col_names=prediction_col,
        pre_data_col_names=pre_cols,
        label_col_names=label_col
)


#Define model context and add data
tru.set_model('Random_Forest')

model_context = ModelOutputContext(model_name = tru._get_current_active_model_name(),
                                     score_type=tru._get_score_type())

print("Adding RF Train Data")
add_data(
    tru.remote_tru,
    rf_train,
    split_name="train",
    column_spec=column_spec,
    model_output_context=model_context
)
rf_train_end = datetime.datetime.now().replace(microsecond=0)
print("RF train ingestion done in, ", rf_train_end-rf_train_start)


while len(tru.get_data_splits())==0:
    time.sleep(2)

#Define column_spec for FIs
tru.set_model('Random_Forest')
rf_fi_start = datetime.datetime.now().replace(microsecond=0)

column_spec_FI = ColumnSpec(
        id_col_name="id",
        feature_influence_col_names = fi_cols
)

model_context_FI = ModelOutputContext(
    model_name = tru._get_current_active_model_name(),
    score_type=tru._get_score_type(),
    influence_type="truera-qii", 
    background_split_name="train"
)

print("Adding RF Train FI data")
add_data(
    tru.remote_tru,
    rf_fi_train,
    split_name="train",
    column_spec=column_spec_FI,
    model_output_context=model_context_FI
)
rf_fi_end = datetime.datetime.now().replace(microsecond=0)
print("RF FI ingestion done in, ", rf_fi_end-rf_fi_start)


rf_prod_all = rf_prod.merge(rf_fi_prod.drop('id', axis=1), left_index=True, right_index=True)

#Add prod data with feature influences
rf_prod_start = datetime.datetime.now().replace(microsecond=0)
#Define column_spec
column_spec_prod = ColumnSpec(
        id_col_name=id_col,
        timestamp_col_name = timestamp_col,
        prediction_col_names=prediction_col,
        pre_data_col_names=pre_cols,
        feature_influence_col_names = fi_cols,
        label_col_names=label_col
)

model_context_prod = ModelOutputContext(
    model_name = tru._get_current_active_model_name(),
    score_type=tru._get_score_type(),
    influence_type="truera-qii", 
    background_split_name="train"
)

print("Adding RF prod data")

tru.add_production_data(
    data=rf_prod_all,
    column_spec=column_spec_prod,
    model_output_context=model_context_prod
)
rf_prod_end = datetime.datetime.now().replace(microsecond=0)
print("RF Prod ingestion done in, ", rf_prod_end-rf_prod_start)


#Add LR data collection
DATA_COLLECTION_NAME="Credit-risk-XGB"

try:
    tru.add_data_collection(DATA_COLLECTION_NAME)
    print('Adding new data colleciton')
except:
    print('Setting existing data collection')
    tru.set_data_collection(DATA_COLLECTION_NAME)
    
#Upload xgboost model
tru.add_model('XGBoost')    


xgb_train_start = datetime.datetime.now().replace(microsecond=0)
#Define column_spec
column_spec = ColumnSpec(
        id_col_name=id_col,
        timestamp_col_name = timestamp_col,
        prediction_col_names=prediction_col,
        pre_data_col_names=pre_cols,
        label_col_names=label_col
)

#Define model context and add data
tru.set_model('XGBoost')

model_context = ModelOutputContext(model_name = tru._get_current_active_model_name(),
                                     score_type=tru._get_score_type())

print("Adding xgboost train data")
add_data(
    tru.remote_tru,
    xgb_train,
    split_name="train",
    column_spec=column_spec,
    model_output_context=model_context
)

xgb_train_end = datetime.datetime.now().replace(microsecond=0)
print("XGB train ingestion done in, ", xgb_train_end-xgb_train_start)

while len(tru.get_data_splits())==0:
    time.sleep(2)

#Add xgboost train FI data
xgb_fi_start = datetime.datetime.now().replace(microsecond=0)

tru.set_model('XGBoost')

column_spec_FI = ColumnSpec(
        id_col_name="id",
        feature_influence_col_names = fi_cols
)

model_context_FI = ModelOutputContext(
    model_name = tru._get_current_active_model_name(),
    score_type=tru._get_score_type(),
    influence_type="truera-qii", 
    background_split_name="train"
)
print("Adding xgboost train FI data")
add_data(
    tru.remote_tru,
    xgb_fi_train,
    split_name="train",
    column_spec=column_spec_FI,
    model_output_context=model_context_FI
)

xgb_fi_end = datetime.datetime.now().replace(microsecond=0)
print("XGB FI ingestion done in, ", xgb_fi_end-xgb_fi_start)

xgb_prod_all = xgb_prod.merge(xgb_fi_prod.drop('id', axis=1), left_index=True, right_index=True)


#Add prod data with feature influences
xgb_prod_start = datetime.datetime.now().replace(microsecond=0)

#Define column_spec
column_spec_prod = ColumnSpec(
        id_col_name=id_col,
        timestamp_col_name = timestamp_col,
        prediction_col_names=prediction_col,
        pre_data_col_names=pre_cols,
        feature_influence_col_names = fi_cols,
        label_col_names=label_col
)

model_context_prod = ModelOutputContext(
    model_name = tru._get_current_active_model_name(),
    score_type=tru._get_score_type(),
    influence_type="truera-qii", 
    background_split_name="train"
)

print("Adding xgboost prod data")

tru.add_production_data(
    data=xgb_prod_all,
    column_spec=column_spec_prod,
    model_output_context=model_context_prod
)
xgb_prod_end = datetime.datetime.now().replace(microsecond=0)
print("XGB Prod ingestion done in, ", xgb_prod_end-xgb_prod_start)

print ("INGESTION DONE:" , datetime.datetime.now().replace(microsecond=0))
ingestion_end = datetime.datetime.now().replace(microsecond=0)

ingestion_time = ingestion_end-ingestion_start
print("Ingestion duration:", ingestion_time)

#Define segment groups and add
grade_segment_defintions = {"grade_a": "grade = 'A'",
 "grade_b": "grade = 'B'",
 "grade_c": "grade = 'C'",
 "grade_d": "grade = 'D'",
 "grades_efg": "grade='E' OR grade='F' OR grade='G'"
                            
}

tru.add_segment_group("grade", grade_segment_defintions)


end_time = datetime.datetime.now().replace(microsecond=0)

print("SCRIPT COMPLETE - TOTAL DURATION:", end_time-start_time)