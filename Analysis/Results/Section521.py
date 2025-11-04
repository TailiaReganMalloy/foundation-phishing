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

f, (ax2, ax3) = plt.subplots(1, 2)
"""
Index(['Subject', 'UserId', 'Experiment', 'Condition', 'Author', 'Style',
       'Feedback', 'Selection', 'Pretraining Accuracy',
       'Training Improvement', 'Training Speed', 'Average Performance',
       'Phishing Experience', 'AI Generation Perception', 'Age', 'Gender',
       'Education', 'Reaction Time', 'Confidence'],
      dtype='object')
"""
exp2 = pd.read_pickle("./Experiment2.pkl")
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

#plt.show()
exp2 = exp2[exp2['Condition'] != 'IBL Selection\nLLM Feedback']

aov = pg.anova(exp2, dv='Pretraining Accuracy', between=['Feedback', 'Selection'])
#print(aov)
aovF = aov[aov['Source'] == 'Feedback']
aovS = aov[aov['Source'] == 'Selection']
aovFS = aov[aov['Source'] == 'Feedback * Selection']

aov = pg.anova(exp2, dv='Pretraining Accuracy', between='Condition')
aov = aov[aov['Source'] == 'Condition']

pretrain = 'To again confirm that the training improvement metric is a valid comparison, we perform an analysis of the pretraining accuracy measure across the feedback, selection, and conditions of experiment 2. To do this we performed a two-way ANOVA comparing the effect of Feedback and Selection on pretraining accuracy. This demonstrated no main effect of Feedback (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) and Selection (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f), and no interaction effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). Then we performed a one-way ANOVA comparing the effect of experiment condition on pretraining accuracy, which also demonstrated no significant effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). These results support our hypothesis that condition would have no effect on pretraining accuracy of experiment 2. For this reason we limit our following analysis to the two metrics of training improvement.' %  (aovF['DF'].item(), aovF['SS'].item(), aovF['F'].item(), aovF['p-unc'].item(), aovF['np2'].item(), aovS['DF'].item(), aovS['SS'].item(), aovS['F'].item(), aovS['p-unc'].item(), aovS['np2'].item(), aovFS['DF'].item(), aovFS['SS'].item(), aovFS['F'].item(), aovFS['p-unc'].item(), aovFS['np2'].item(), aov['ddof1'].item(), aov['ddof2'].item(), aov['F'].item(), aov['p-unc'].item(), aov['np2'].item())

#print(pretrain)

aov = pg.anova(data=exp2, dv='Training Improvement', between='Condition')
pt = pg.pairwise_tukey(data=exp2, dv='Training Improvement', between='Condition')
ptSig = pt[pt['p-tukey'] < 0.05 ]
ptSig1 = ptSig.iloc[0]
#ptSig2 = ptSig.iloc[1]

improvement = '\\textbf{Training Improvement:} We are interested in determining which of the conditions is the most challenging learning task so that we can use exclusively those emails in experiment 2 in our comparison of appraoches to improve training outcomes. To that end, we investigated the impact of experiment 2 codition on training improvement. A one-way ANOVA was performed to compare the effect of experiment 2 condition on training improvement revealed that there was a statistically significant difference in training improvement between at least two conditions (F(%.3f, %.3f)=%.3f, p=%.3f). Tukey\'s HSD Test for multiple comparisons found that the mean value of training improvement was significantly higher in the IBL+LLM Selection IBL+LLM Feedback condition compared to the the Random Selection Points Feedback condition (p=%.3f, diff=%.3f se=%.3f, T=%.3f, hedges=%.3f) . There was no statistically significant difference between each other comparison of experiment conditions. From these results we can conclude that the IBL+LLM Selection IBL+LLM Feedback condition resulted in the largest improvement in training performance of all the conditions in experiment 2.' % (aov['ddof1'].item(), aov['ddof2'].item(), aov['F'].item(), aov['p-unc'].item(), ptSig1['p-tukey'].item(), ptSig1['diff'].item(), ptSig1['se'].item(), ptSig1['T'].item(), ptSig1['hedges'].item())

#print(improvement)

aov = pg.anova(data=exp2, dv='Training Speed', between='Condition')
pt = pg.pairwise_tukey(data=exp2, dv='Training Speed', between='Condition')
ptSig = pt[pt['p-tukey'] < 0.05 ]
ptSig1 = ptSig.iloc[0]
#ptSig2 = ptSig.iloc[1]

speed = '\\textbf{Training Speed:} While the proposed IBL+LLM Selection IBL+LLM Feedback method demonstrated the highest training improvement, there is a question of whether or not selecting emails to be challenging for participants slows down training speed. To determine this, a one-way ANOVA was performed to compare the effect of experiment 2 condition on training improvement revealed that there was a statistically significant difference in training improvement between at least two conditions (F(%.3f, %.3f)=%.3f, p=%.3f). Tukey\'s HSD Test for multiple comparisons found that the mean value of training improvement was significantly higher in the IBL+LLM Selection IBL+LLM Feedback condition compared to the the Random Selection Points Feedback condition (p=%.3f, diff=%.3f se=%.3f, T=%.3f, hedges=%.3f) . There was no statistically significant difference between each other comparison of experiment conditions. From these results we can conclude that the IBL+LLM Selection IBL+LLM Feedback condition resulted in the largest improvement in training performance of all the conditions in experiment 2.' % (aov['ddof1'].item(), aov['ddof2'].item(), aov['F'].item(), aov['p-unc'].item(), ptSig1['p-tukey'].item(), ptSig1['diff'].item(), ptSig1['se'].item(), ptSig1['T'].item(), ptSig1['hedges'].item())

print(speed)