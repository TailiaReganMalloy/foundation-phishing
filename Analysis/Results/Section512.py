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
#print(exp1.columns)
"""
Index(['Subject', 'UserId', 'Experiment', 'Condition', 'Author', 'Style',
       'Author', 'Style', 'Pretraining Accuracy', 'Training Improvement',
       'Training Speed', 'Average Performance', 'Phishing Experience',
       'AI Generation Perception', 'Age', 'Gender', 'Education',
       'Reaction Time', 'Confidence'],
      dtype='object')
"""


aovF = aov = pg.anova(data=exp1, dv='Average Performance', between=['AI Generation Perception', 'Author'])
#print(aovF)
aovFA = aovF[aovF['Source'] == 'Author']
aovFF = aovF[aovF['Source'] == 'AI Generation Perception']
aovFI = aovF[aovF['Source'] == 'AI Generation Perception * Author']

aovP = pg.anova(data=exp1[exp1['Author'] == 'Human'], dv='Average Performance', between='AI Generation Perception')
#print(aovP)

aovI = pg.anova(data=exp1[exp1['Author'] == 'LLM'], dv='Average Performance', between='AI Generation Perception')
#print(aovI)

perception = '\\textbf{AI Generation Perception}:  We are interested in the relative impact of AI Generation Perception on average performance between different types of Author, to see whether recieving Author from an AI chatbot reduces the effect observed in experiment 1. We first preformed a two-way ANOVA to compare the effect of AI Generation Perception and Author on Average Performance, this revealed a sigificant effect of AI Generation Perception (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) and Author (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f), as well as an interaction effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). After this, we performed two seperate one-way ANOVAs on the effect of AI Generation Perception on Average Performance, limiting the analysis to the Points Author conditions and the IBL+LLM Author conditions seperately. This analysis revealed a significant effect of AI Generation Perception on Average performance in the Points Author conditions (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) but not in the IBL+LLM conditions (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f).' % (aovFA['DF'].item(), aovFA['SS'].item(), aovFA['F'].item(), aovFA['p-unc'].item(), aovFA['np2'].item(), aovFF['DF'].item(), aovFF['SS'].item(), aovFF['F'].item(), aovFF['p-unc'].item(), aovFF['np2'].item(), aovFI['DF'].item(), aovFI['SS'].item(), aovFI['F'].item(), aovFI['p-unc'].item(), aovFI['np2'].item(), aovP['ddof1'].item(), aovP['ddof2'].item(), aovP['F'].item(), aovP['p-unc'].item(), aovP['np2'].item(), aovI['ddof1'].item(), aovI['ddof2'].item(), aovI['F'].item(), aovI['p-unc'].item(), aovI['np2'].item())

print(perception)

aovF = pg.anova(data=exp1, dv='Average Performance', between=['Phishing Experience', 'Author'])
#print(aovF)
aovFA = aovF[aovF['Source'] == 'Phishing Experience']
aovFF = aovF[aovF['Source'] == 'Phishing Experience']
aovFI = aovF[aovF['Source'] == 'Phishing Experience * Author']

aovP = pg.anova(data=exp1[exp1['Style'] == 'Plain'], dv='Average Performance', between='Phishing Experience')
#print(aovP)

aovI = pg.anova(data=exp1[exp1['Style'] == 'LLM'], dv='Average Performance', between='Phishing Experience')
#print(aovI)

experience = '\\textbf{Phishing Experience}:  We are interested in the relative impact of Phishing Experience on average performance between different types of Style, to see whether recieving Style from an AI chatbot reduces the effect observed in experiment 1. We first preformed a two-way ANOVA to compare the effect of Phishing Experience and Style on Average Performance, this revealed a sigificant effect of Phishing Experience (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) and Style (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f), as well as an interaction effect (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f). After this, we performed two seperate one-way ANOVAs on the effect of Phishing Experience on Average Performance, limiting the analysis to the Points Style conditions and the IBL+LLM Style conditions seperately. This analysis revealed a significant effect of Phishing Experience on Average performance in the Points Style conditions (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f) but not in the IBL+LLM conditions (F(%.3f,%.3f)=%.3f, p=%.3f, $\\eta^2$=%.3f).' % (aovFA['DF'].item(), aovFA['SS'].item(), aovFA['F'].item(), aovFA['p-unc'].item(), aovFA['np2'].item(), aovFF['DF'].item(), aovFF['SS'].item(), aovFF['F'].item(), aovFF['p-unc'].item(), aovFF['np2'].item(), aovFI['DF'].item(), aovFI['SS'].item(), aovFI['F'].item(), aovFI['p-unc'].item(), aovFI['np2'].item(), aovP['ddof1'].item(), aovP['ddof2'].item(), aovP['F'].item(), aovP['p-unc'].item(), aovP['np2'].item(), aovI['ddof1'].item(), aovI['ddof2'].item(), aovI['F'].item(), aovI['p-unc'].item(), aovI['np2'].item())

print(experience)
