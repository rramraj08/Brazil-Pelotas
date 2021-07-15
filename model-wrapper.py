
import sys
import cdsw
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

# == for Testing ==
features = ['stunting_2cat','agemonr1','chsexr1','agegapr1','momeduyrsr1','hhsizer1','wi_newr1']

args={"stunting_2cat":"1",
      "agemonr1":"49",
      "chsexr1":"1",
      "agegapr1":"6",
      "momeduyrsr1":"5",
      "hhsizer1":"5",
      "wi_newr1":"0.11"}


#==Main Funtion==

def PredictFunc(args):
  # Load Data
  filtArgs={key:[args[key]] for key in features}
  data=pd.DataFrame.from_dict(filtArgs)
  
  # Load Model
  with open('HeightPredictor.pickle','rb') as handle:
    mdl=pickle.load(handle)
    model=pickle.loads(mdl)
    
    
  # Get Prediction
  prediction=model.predict(data)
  
  #Return Prediction
  return prediction