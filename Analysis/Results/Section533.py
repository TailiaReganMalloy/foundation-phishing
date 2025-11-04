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
"""
Index(['Subject', 'UserId', 'Experiment', 'Condition', 'Author', 'Style',
       'Feedback', 'Selection', 'Pretraining Accuracy', 'Training Improvement',
       'Training Speed', 'Average Performance', 'Phishing Experience',
       'AI Generation Perception', 'Age', 'Gender', 'Education',
       'Reaction Time', 'Confidence'],
      dtype='object')
"""

exp21 = exp2[exp2['Condition'].isin(['IBL+LLM Selection\nIBL+LLM Feedback', 'IBL Selection\nLLM Feedback'])]

aovP = pg.anova(data=exp21, dv='Pretraining Accuracy', between='Condition')
#print(aovP)

aovTi = pg.anova(data=exp21, dv='Training Improvement', between='Condition')
exp22 = exp2[exp2['Condition'].isin(['Random Selection\nIBL+LLM Feedback', 'IBL Selection\nLLM Feedback'])]

aov = pg.anova(data=exp22, dv='Pretraining Accuracy', between='Condition')
#print(aov)

aovTs = pg.anova(data=exp22, dv='Training Speed', between='Condition')
#print(aovTs)

pt = pg.pairwise_tukey(data=exp2, dv='Average Performance', between='Condition')
ptSig = pt[pt['p-tukey'] < 0.05 ]
ptSig1 = ptSig.iloc[0]

pt = pg.pairwise_tukey(data=exp2, dv='Training Improvement', between='Condition')
ptSig = pt[pt['p-tukey'] < 0.05 ]
ptSig2 = ptSig.iloc[0]

ablation = 'The final set of analysis is done to compare the importance of the integration of the IBL and LLM model with an alternative condtion that additionally selects example emails and provides natural language feedback, but without the IBL-LLM connection. This ablation condition selected emails using an IBL model based on email features, not email embeddings, and additionally provided natural language feedback using an LLM, but without prompting information from the IBL model. This ablation condition is referred to as the IBL Selection LLM Feedback condition. To this end, the following statistical analysis adds the ablation condition to the analysis of experiment 2 conditions. To again confirm that the training improvement metric is a valid comparison, to do this we performed a one-way ANOVA comparing the effect of experiment condition on pretraining accuracy, which demonstrated no significant effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). These results support our hypothesis that condition would have no effect on pretraining accuracy between the ablation condition and the full condition. For this reason we limit our following analysis to the two metrics of training improvement. A one-way ANOVA compared the effect of experiment condition on training improveemnt and demonstrated a significant variation (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). A follow-up Tukey HSD demonstrated that the Random Selection IBL+LLM Feedback had a faster training speed compared to the IBL Selection LLM Feedback condition (p=%.3f, diff=%.3f se=%.3f, T=%.3f, hedges=%.3f), while no other condition had a significant variation. This makes intuitive sense as the IBL+LLM and IBL email selection method intentionally choose difficult emails for participants. However, the goal of this selection is that it allows for more significant improvements in training. To compare this effect in the ablation experiment, we next compared the effect of condition on training improvement. A one-way ANOVA demonstrated a significant variation of  the effect of experiment condition on training improvement (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). A follow-up Tukey HSD demonstrated that the IBL+LLM Selection and IBL+LLM Feedback condition had significantly higher training improvement compared to the IBL Selection LLM Feedback condition (p=%.3f, diff=%.3f se=%.3f, T=%.3f, hedges=%.3f). Overall, these results demonstrate that the IBL Selection LLM Feedback ablation condition does not result in either the higher training improvement of the IBL+LLM Selection IBL+LLM Feedback condition, nor the faster training speed of the Random Selection IBL+LLM Feedback condition. From this we can conclude that the connection of the IBL and LLM models is crucial for achieving the benefits of our proposed training method.' %  (aovP['ddof1'].item(), aovP['ddof2'].item(), aovP['F'].item(), aovP['p-unc'].item(), aovP['np2'].item(),  aovTi['ddof1'].item(), aovTi['ddof2'].item(), aovTi['F'].item(), aovTi['p-unc'].item(), aovTi['np2'].item(), ptSig1['p-tukey'].item(), ptSig1['diff'].item(), ptSig1['se'].item(), ptSig1['T'].item(), ptSig1['hedges'].item(),  aovTs['ddof1'].item(), aovTs['ddof2'].item(), aovTs['F'].item(), aovTs['p-unc'].item(), aovTs['np2'].item(), ptSig2['p-tukey'].item(), ptSig2['diff'].item(), ptSig2['se'].item(), ptSig2['T'].item(), ptSig2['hedges'].item())

print(ablation)