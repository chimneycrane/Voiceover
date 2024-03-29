import os
import subprocess
import pandas as pd
import xgboost as xgb
from ExtractFeatures import *

def predict(audio_name, work_dir):
    wd = os.getcwd()
    Extract(audio_name, work_dir)
    features_to_use = ["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx"]

    test_df = pd.read_csv(work_dir+'/Features.csv')
    
    test_X = test_df[features_to_use]
    xgtest = xgb.DMatrix(test_X)

    model = xgb.Booster({'nthread':4})
    model.load_model(wd+"/voice-classify.model")

    pred_test_y = model.predict(xgtest)

    if pred_test_y >= 0.5:
        return "male"
    return "female"
