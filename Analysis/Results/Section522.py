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
#print(exp2.columns)
"""
Index(['Subject', 'UserId', 'Experiment', 'Condition', 'Author', 'Style',
       'Feedback', 'Selection', 'Pretraining Accuracy', 'Training Improvement',
       'Training Speed', 'Average Performance', 'Phishing Experience',
       'AI Generation Perception', 'Age', 'Gender', 'Education',
       'Reaction Time', 'Confidence'],
      dtype='object')
"""


aovF = aov = pg.anova(data=exp2, dv='Average Performance', between=['AI Generation Perception', 'Feedback'])
#print(aovF)
aovFA = aovF[aovF['Source'] == 'AI Generation Perception']
aovFF = aovF[aovF['Source'] == 'AI Generation Perception']
aovFI = aovF[aovF['Source'] == 'AI Generation Perception * Feedback']

aovP = pg.anova(data=exp2[exp2['Feedback'] == 'Points'], dv='Average Performance', between='AI Generation Perception')
#print(aovP)

aovI = pg.anova(data=exp2[exp2['Feedback'] == 'IBL+LLM'], dv='Average Performance', between='AI Generation Perception')
#print(aovI)

perception = '\\textbf{AI Generation Perception}:  We are interested in the relative impact of AI Generation Perception on average performance between different types of feedback, to see whether recieving feedback from an AI chatbot reduces the effect observed in experiment 1. We first preformed a two-way ANOVA to compare the effect of AI Generation Perception and Feedback on Average Performance, this revealed a sigificant effect of AI Generation Perception (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) and Feedback (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f), as well as an interaction effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). After this, we performed two seperate one-way ANOVAs on the effect of AI Generation Perception on Average Performance, limiting the analysis to the Points Feedback conditions and the IBL+LLM Feedback conditions seperately. This analysis revealed a significant effect of AI Generation Perception on Average performance in the Points Feedback conditions (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) but not in the IBL+LLM conditions (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f).' % (aovFA['DF'].item(), aovFA['SS'].item(), aovFA['F'].item(), aovFA['p-unc'].item(), aovFA['np2'].item(), aovFF['DF'].item(), aovFF['SS'].item(), aovFF['F'].item(), aovFF['p-unc'].item(), aovFF['np2'].item(), aovFI['DF'].item(), aovFI['SS'].item(), aovFI['F'].item(), aovFI['p-unc'].item(), aovFI['np2'].item(), aovP['ddof1'].item(), aovP['ddof2'].item(), aovP['F'].item(), aovP['p-unc'].item(), aovP['np2'].item(), aovI['ddof1'].item(), aovI['ddof2'].item(), aovI['F'].item(), aovI['p-unc'].item(), aovI['np2'].item())

#print(perception)

aovF = aov = pg.anova(data=exp2, dv='Average Performance', between=['Phishing Experience', 'Feedback'])
#print(aovF)
aovFA = aovF[aovF['Source'] == 'Phishing Experience']
aovFF = aovF[aovF['Source'] == 'Phishing Experience']
aovFI = aovF[aovF['Source'] == 'Phishing Experience * Feedback']

aovP = pg.anova(data=exp2[exp2['Selection'] == 'Random'], dv='Average Performance', between='Phishing Experience')
#print(aovP)

aovI = pg.anova(data=exp2[exp2['Selection'] == 'IBL+LLM'], dv='Average Performance', between='Phishing Experience')
#print(aovI)

experience = '\\textbf{Phishing Experience}:  We are interested in the relative impact of Phishing Experience on average performance between different types of Selection, to see whether recieving Selection from an AI chatbot reduces the effect observed in experiment 1. We first preformed a two-way ANOVA to compare the effect of Phishing Experience and Selection on Average Performance, this revealed a sigificant effect of Phishing Experience (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) and Selection (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f), as well as an interaction effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). After this, we performed two seperate one-way ANOVAs on the effect of Phishing Experience on Average Performance, limiting the analysis to the Points Selection conditions and the IBL+LLM Selection conditions seperately. This analysis revealed a significant effect of Phishing Experience on Average performance in the Points Selection conditions (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) but not in the IBL+LLM conditions (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f).' % (aovFA['DF'].item(), aovFA['SS'].item(), aovFA['F'].item(), aovFA['p-unc'].item(), aovFA['np2'].item(), aovFF['DF'].item(), aovFF['SS'].item(), aovFF['F'].item(), aovFF['p-unc'].item(), aovFF['np2'].item(), aovFI['DF'].item(), aovFI['SS'].item(), aovFI['F'].item(), aovFI['p-unc'].item(), aovFI['np2'].item(), aovP['ddof1'].item(), aovP['ddof2'].item(), aovP['F'].item(), aovP['p-unc'].item(), aovP['np2'].item(), aovI['ddof1'].item(), aovI['ddof2'].item(), aovI['F'].item(), aovI['p-unc'].item(), aovI['np2'].item())

print(experience)
