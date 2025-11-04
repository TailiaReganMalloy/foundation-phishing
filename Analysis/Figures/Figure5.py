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
print(pdf.columns)
"""
Index(['Subject', 'UserId', 'Experiment', 'Condition', 'Author', 'Style',
       'Feedback', 'Selection', 'Pretraining Accuracy',
       'Training Improvement', 'Training Speed', 'Average Performance',
       'Phishing Experience', 'AI Generation Perception', 'Age', 'Gender',
       'Education', 'Reaction Time', 'Confidence'],
      dtype='object')
"""
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

exp1 = pdf[pdf['Experiment'] == 1]
"""
['Human Written\nPlain Styled' 'LLM Written\nLLM Styled'
 'Human Written\nLLM Styled' 'LLM Written\nPlain Styled']
"""
# AI Generation Perception 
aig = exp1.groupby(['AI Generation Perception'], as_index=False)['Average Performance'].mean()
g = sns.regplot(exp1, x='AI Generation Perception', y='Average Performance', ax=ax1, scatter_kws={'alpha':0.25})
r, p = sp.stats.pearsonr(exp1['AI Generation Perception'], exp1['Average Performance'])
g.text(.05, .05, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax1.transAxes, fontsize=14)

aig = exp1.groupby(['Phishing Experience'], as_index=False)['Average Performance'].mean()
g = sns.regplot(exp1, x='Phishing Experience', y='Average Performance', ax=ax2, scatter_kws={'alpha':0.25})
r, p = sp.stats.pearsonr(exp1['Phishing Experience'], exp1['Average Performance'])
g.text(.05, .05, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax2.transAxes, fontsize=14)

ax2.set_ylabel("")
ax1.set_ylabel("Average Performance", fontsize=14)

ax1.set_xlabel("AI Generation Perception", fontsize=14)
ax2.set_xlabel("Phishing Experience", fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.tick_params(axis='both', which='minor', labelsize=10)

ax2.tick_params(axis='x', which='major', labelsize=12)
ax2.tick_params(axis='x', which='minor', labelsize=10)

ax1.set_title("Average Performance by\nAI Generation Perception", fontsize=14)
ax2.set_title("Average Performance by\nPhishing Experience", fontsize=14)

f.set_size_inches(10, 5)
plt.subplots_adjust(left=0.065, bottom=0.115, right=0.99, top=0.9, wspace=0.15, hspace=None)
plt.savefig('Images/Figure5.png')

plt.show()