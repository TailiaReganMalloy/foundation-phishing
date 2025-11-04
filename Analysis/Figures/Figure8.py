import warnings
warnings.filterwarnings("ignore")

import statsmodels
import pingouin as pg 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import scipy as sp

from statsmodels.tools.sm_exceptions import ValueWarning
warnings.simplefilter('ignore', ValueWarning)

exp2 = pd.read_pickle("./Experiment2.pkl")
#print(pdf.columns)
"""
Index(['Subject', 'UserId', 'Experiment', 'Condition', 'Author', 'Style',
       'Feedback', 'Feedback', 'Pretraining Accuracy',
       'Training Improvement', 'Training Speed', 'Average Performance',
       'AI Generation Perception', 'AI Generation Perception', 'Age', 'Gender',
       'Education', 'Reaction Time', 'Confidence'],
      dtype='object')
"""

print(exp2['Condition'].unique())

