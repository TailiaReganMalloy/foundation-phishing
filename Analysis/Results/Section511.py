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

df = pd.read_pickle("../Data/Clean/ParticipantData.pkl")
#print(df.columns)
"""
Index(['UserId', 'Experiment', 'Condition', 'Author', 'Style', 'Feedback',
       'Selection', 'PhaseValue', 'PhaseTrial', 'ExperimentTrial', 'DataType',
       'EmailId', 'EmailType', 'Decision', 'Confidence', 'EmailAction',
       'ReactionTime', 'Correct', 'MessageNum', 'Message', 'Age', 'Gender',
       'Education', 'Country', 'Victim', 'Chatbot', 'Q0', 'Q1', 'Q2', 'Q3',
       'Q4', 'Q5', 'PQ0', 'PQ1', 'PQ2', 'PQ3', 'PQ4', 'PQ5'],
      dtype='object')
"""

rdf = df[df['DataType'] == 'Response']
exp1 = rdf[rdf['Experiment'] == 1]

aov = pg.mixed_anova(data=exp1, dv='Correct', between='Condition', within='ExperimentTrial', subject='UserId').round(3)
aovC = aov[aov['Source'] == 'Condition']
aovE = aov[aov['Source'] == 'ExperimentTrial']
aovI = aov[aov['Source'] == 'Interaction']

intro = '\\subsubsection{Comparing Performance Measures by Condition} \n In comparing the performance of participants by condition we initially investigated whether or not there was a significant difference between conditions of participant categorization accuracy at the trial level. To assess this, we performed a repeated measures MANOVA assessing participant categorization accuracy between conditions and within trials of the experiment. This analysis demonstrated no significant main effect of condition (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) nor trial (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) nor an interaction effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). This result was unsurprising as our analysis is primarily concerned with the difference between pretraining and posttraining performance in participants. This is the reason that our stated hypotheses are in reference to the difference between pretraining and posttraining performance, instead of performance within trials of the experiment. Because of this, we limit our analysis to measures of performance that are connected to training outcomes.' % (aovC['DF1'], aovC['DF2'], aovC['F'], aovC['p-unc'], aovC['np2'], aovE['DF1'], aovE['DF2'], aovE['F'], aovE['p-unc'], aovE['np2'], aovE['DF1'], aovE['DF2'], aovE['F'], aovE['p-unc'], aovE['np2'])

print(intro)

pdf = pd.read_pickle("./Results.pkl")
exp1 = pdf[pdf['Experiment'] == 1]
aovC = pg.anova(data=exp1, dv='Pretraining Accuracy', between='Condition')
aovAS = pg.anova(data=exp1, dv='Pretraining Accuracy', between=['Author', 'Style'])
aovASA = aovAS[aovAS['Source'] == 'Author']
aovASS = aovAS[aovAS['Source'] == 'Style']
aovASI = aovAS[aovAS['Source'] == 'Author * Style']

pretrain = '\\textbf{Pretraining Accuracy:} Before performing our comparison of conditions in terms of the difficulty of training, we needed to ensure that the training improvement metric is a valid comparison. This is due to the potential impact that significantly different pretraining accuracies would have on the validity of comparing conditions based on training improvement. We first preformed a two-way ANOVA to compare the effect of Author and Style on pretraining accuracy, this revealed a sigificant effect of Author (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) and Style (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f), as well as an interaction effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). For this reason, we limit our comparison of experiment 1 training to the training improvement and training speed measure.' % (aovASA['DF'].item(), aovASA['SS'].item(), aovASA['F'].item(), aovASA['p-unc'].item(), aovASA['np2'].item(), aovASS['DF'].item(), aovASS['SS'].item(), aovASS['F'].item(), aovASS['p-unc'].item(), aovASS['np2'].item(), aovASI['DF'].item(), aovASI['SS'].item(), aovASI['F'].item(), aovASI['p-unc'].item(), aovASI['np2'].item())

print(pretrain)

aov = pg.anova(data=exp1, dv='Categorization Improvement', between='Condition')
pt = pg.pairwise_tukey(data=exp1, dv='Categorization Improvement', between='Condition')
ptSig = pt[pt['p-tukey'] < 0.05 ]
ptSig1 = ptSig.iloc[0]
ptSig2 = ptSig.iloc[1]

improvement = '\\textbf{Training Improvement:} We are interested in determining which of the conditions is the most challenging learning task so that we can use exclusively those emails in experiment 2 in our comparison of appraoches to improve training outcomes. To that end, we investigated the impact of experiment 1 codition on categorization improvement. A one-way ANOVA was performed to compare the effect of experiment 1 condition on categorization improvement revealed that there was a statistically significant difference in categorization improvement between at least two conditions (F(%.3f, %.3f)=%.3f, p=%.3f). Tukey\'s HSD Test for multiple comparisons found that the mean value of categorization improvement was significantly lower in the human written GPT-4 styled condition compared to the the GPT-4 written GPT-4 styled condition (p=%.3f, diff=%.3f se=%.3f, T=%.3f, hedges=%.3f) and was significantly lower in the human written GPT-4 styled condition compared to the GPT-4 written GPT-4 styled condition (p = %.3f, diff=%.3f se=%.3f, T=%.3f, hedges=%.3f). There was no statistically significant difference between each other comparison of experiment conditions. From these results we can conclude that the human written GPT-4 styled condition is the most challenging from the perspective of categorization improvement.' % (aov['ddof1'].item(), aov['ddof2'].item(), aov['F'].item(), aov['p-unc'].item(), ptSig1['p-tukey'].item(), ptSig1['diff'].item(), ptSig1['se'].item(), ptSig1['T'].item(), ptSig1['hedges'].item(), ptSig1['p-tukey'].item(), ptSig2['diff'].item(), ptSig2['se'].item(), ptSig2['T'].item(), ptSig2['hedges'].item())

print(improvement)

aov = pg.anova(data=exp1, dv='Training Speed', between='Condition')
pt = pg.pairwise_tukey(data=exp1, dv='Training Speed', between='Condition')
ptSig = pt[pt['p-tukey'] < 1 ]
ptSig1 = ptSig.iloc[0]
ptSig2 = ptSig.iloc[1]

speed = '\\textbf{Training Speed:} To get a better sense of the overall difficulty of learning in each condition beyond training improvement, we next compared conditions in terms of the speed of learning. A one-way ANOVA was performed to compare the effect of experiment 1 condition on training speed revealed that there was a statistically significant difference in training speed between at least two conditions (F(%.3f, %.3f)=%.3f, p=%.3f).  Tukey\'s HSD Test for multiple comparisons found that the mean value of training speed was significantly lower in the human written LLM styled condition compared to the the LLM written LLM styled condition (p=%.3f, diff=%.3f se=%.3f, T=%.3f, hedges=%.3f) and was significantly lower in the LLM written Plain styled condition compared to the LLM written LLM styled condition (p=%.3f, diff=%.3f se=%.3f, T=%.3f, hedges=%.3f). There was no statistically significant difference between each other comparison of experiment conditions. From this analysis of the differences in training speed across conditions we can see that the human written LLM styled condition is significantly more difficult in terms of both training improvement and training speed. This motivates our use of human written LLM styled emails in experiment 2, to provide for the best method of determining the effect of the training improvement approaches we propose.'  % (aov['ddof1'].item(), aov['ddof2'].item(), aov['F'].item(), aov['p-unc'].item(), ptSig1['p-tukey'].item(), ptSig1['diff'].item(), ptSig1['se'].item(), ptSig1['T'].item(), ptSig1['hedges'].item(), ptSig2['p-tukey'].item(), ptSig2['diff'].item(), ptSig2['se'].item(), ptSig2['T'].item(), ptSig2['hedges'].item())

print(speed)