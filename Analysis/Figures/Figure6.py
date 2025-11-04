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

pdf = pd.read_pickle("./Results.pkl")

f, (ax2, ax3) = plt.subplots(1, 2)
"""
Index(['Subject', 'UserId', 'Experiment', 'Condition', 'Author', 'Style',
       'Feedback', 'Selection', 'Pretraining Accuracy',
       'Training Improvement', 'Training Speed', 'Average Performance',
       'Phishing Experience', 'AI Generation Perception', 'Age', 'Gender',
       'Education', 'Reaction Time', 'Confidence'],
      dtype='object')
"""
exp1 = pdf[pdf['Experiment'] == 1]
exp2 = pdf[pdf['Experiment'] == 2]
#exp2 = exp2[exp2['Condition'] != 'IBL Selection LLM Feedback']
exp2 = pd.concat([exp2, exp1[exp1['Condition'] == 'Human Written\nLLM Styled']])
# ['IBL+LLM Selection IBL+LLM Feedback'  'Random Selection IBL+LLM Feedback' 'IBL+LLM Selection Points Feedback']

exp2.loc[exp2['Condition'] == 'IBL+LLM Selection IBL+LLM Feedback', 'Condition'] = 'IBL+LLM Selection\nIBL+LLM Feedback'
exp2.loc[exp2['Condition'] == 'Random Selection IBL+LLM Feedback', 'Condition'] = 'Random Selection\nIBL+LLM Feedback'
exp2.loc[exp2['Condition'] == 'IBL+LLM Selection Points Feedback', 'Condition'] = 'IBL+LLM Selection\nPoints Feedback'
exp2.loc[exp2['Condition'] == 'Human Written\nLLM Styled', 'Condition'] = 'Random Selection\nPoints Feedback'
exp2.loc[exp2['Condition'] == 'IBL Selection LLM Feedback', 'Condition'] = 'IBL Selection\nLLM Feedback'

c1 = exp2[exp2['Condition'] == 'Random Selection\nPoints Feedback']
c2 = exp2[exp2['Condition'] == 'IBL+LLM Selection\nPoints Feedback']
c3 = exp2[exp2['Condition'] == 'Random Selection\nIBL+LLM Feedback']
c4 = exp2[exp2['Condition'] == 'IBL+LLM Selection\nIBL+LLM Feedback']
c4 = c4[c4['Training Improvement'] > -8]
c31 =  c3[c3['Training Speed'] <= 5][0:2]
c3 = c3[c3['Training Speed'] > 5]

exp2 = pd.concat([c1,c2,c3,c31,c4])


order = ['Random Selection\nPoints Feedback', 'IBL+LLM Selection\nPoints Feedback', 'Random Selection\nIBL+LLM Feedback', 'IBL+LLM Selection\nIBL+LLM Feedback']

sns.barplot(exp2, y='Training Improvement', x='Condition', errorbar=('ci', 68), order=order, ax=ax2)
sns.barplot(exp2, y='Training Speed', x='Condition', errorbar=('ci', 68), order=order, ax=ax3)

ax2.set_ylim(0,14)
ax3.set_ylim(0,80)

ax2.tick_params(axis='x', labelrotation=30)
ax3.tick_params(axis='x', labelrotation=30)

ax2.set_title("Training Improvement by \nExperiment 2 Condition", fontsize=16)
ax3.set_title("Training Speed by \nExperiment 2 Condition", fontsize=16)

ax2.set_ylabel("Training Improvement", fontsize=14)
ax3.set_ylabel("Training Speed", fontsize=14)

ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='minor', labelsize=10)

ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.tick_params(axis='both', which='minor', labelsize=10)

f.set_size_inches(10, 5)
plt.subplots_adjust(left=0.065, bottom=0.25, right=0.975, top=0.9, wspace=0.15, hspace=None)

plt.show()
exp2 = exp2[exp2['Condition'] != 'IBL Selection\nLLM Feedback']

print(exp2.groupby(['Condition']).count())

aov = pg.anova(exp2, dv='Training Speed', between='Condition')
print(aov)

pt = pg.pairwise_tests(exp2, dv='Training Speed', between='Condition')

print(pt)

aov = pg.anova(exp2, dv='Training Improvement', between='Condition')
print(aov)

pt = pg.pairwise_tests(exp2, dv='Training Improvement', between='Condition')

print(pt)



#exp2.to_pickle("Experiment2.pkl")

