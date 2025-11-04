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

exp1 = pd.read_pickle("./Results.pkl")
exp1 = exp1[exp1['Experiment'] == 1]
#print(pdf.columns)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['AI Generation Perception'], data['Average Performance'])
    ax = plt.gca()
    ax.text(.1, .05, 'r={:.8f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)

#exp2["AI Generation Perception"] = exp2["AI Generation Perception"] ** 10
g = sns.lmplot(
    data=exp1, x="AI Generation Perception", y="Average Performance", col="Author", height=4,
)

g.map_dataframe(annotate)

a1 = g.axes[0,0]
a1.set_title("Human Authored Email Conditions\nPerformance by AI Generation Perception")
a2 = g.axes[0,1]
a2.set_title("LLM Authored Email Conditions\nPerformance by AI Generation Perception")
plt.show()


