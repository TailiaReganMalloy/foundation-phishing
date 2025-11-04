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
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['AI Generation Perception'], data['Average Performance'])
    ax = plt.gca()
    ax.text(.1, .05, 'r={:.8f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)

#exp2["AI Generation Perception"] = exp2["AI Generation Perception"] ** 10
g = sns.lmplot(
    data=exp2, x="AI Generation Perception", y="Average Performance", col="Feedback", height=4,
)

g.map_dataframe(annotate)

a1 = g.axes[0,0]
a1.set_title("Point Feedback Conditions\nPerformance by AI Generation Perception")
a2 = g.axes[0,1]
a2.set_title("IBL+LLM Feedback Conditions\nPerformance by AI Generation Perception")
a2.set_xlim(0.2,1)
plt.show()


