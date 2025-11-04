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

df = pd.read_pickle("../Data/Cleaned/ParticipantData.pkl")
"""
Index(['UserId', 'Experiment', 'Condition', 'PhaseValue', 'PhaseTrial',
       'ExperimentTrial', 'DataType', 'EmailId', 'EmailType', 'Decision',
       'Confidence', 'EmailAction', 'ReactionTime', 'Correct', 'MessageNum',
       'Message', 'Age', 'Gender', 'Education', 'Country', 'Victim', 'Chatbot',
       'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'PQ0', 'PQ1', 'PQ2', 'PQ3', 'PQ4',
       'PQ5'],
"""
df = df.dropna()
df = df[df['PhaseValue'] == 'Training']

exp1 = df[df['Experiment'] == 1]
mixed_anova = pg.mixed_anova(data=exp1, dv='Correct', between='Condition',  within='PhaseTrial', subject='UserId')
#print(mixed_anova)
"""
\begin{tabular}{lrrrrrrrrrlrr}
\toprule
Source & SS & DF1 & DF2 & MS & F & p-unc & p-GG-corr & np2 & eps & sphericity & W-spher & p-spher \\
\midrule
Condition & 2.560246 & 3 & 204 & 0.853415 & 0.829578 & 0.478967 & NaN & 0.012053 & NaN & NaN & NaN & NaN \\
PhaseTrial & 5.772957 & 39 & 7956 & 0.148025 & 0.972538 & 0.518704 & 0.512240 & 0.004745 & 0.817006 & False & 0.010281 & 0.004749 \\
Interaction & 18.914661 & 117 & 7956 & 0.161664 & 1.062150 & 0.306874 & NaN & 0.015380 & NaN & NaN & NaN & NaN \\
\bottomrule
\end{tabular}
"""

cond4 = exp1[exp1['Condition'] == 'Human Written LLM Styled']
exp2 = df[df['Experiment'] == 2]
exp2 = pd.concat([cond4, exp2], ignore_index=True)

# Add a column on whether emails were selected by IBL+LLM, IBL, or points
exp2['Selection'] = "None"
exp2['Feedback'] = "None"
#print(exp2['Condition'].unique())

exp2.loc[exp2['Condition'] == "Human Written LLM Styled", 'Selection'] = "Random"
exp2.loc[exp2['Condition'] == "IBL+LLM Selection Points Feedback", 'Selection'] = "IBL+LLM"
exp2.loc[exp2['Condition'] == "IBL+LLM Selection IBL+LLM Feedback", 'Selection'] = "IBL+LLM"
exp2.loc[exp2['Condition'] == "Random Selection IBL+LLM Feedbac", 'Selection'] = "Random"
exp2.loc[exp2['Condition'] == "IBL Selection LLM Feedback", 'Selection'] = "IBL"

exp2.loc[exp2['Condition'] == "Human Written LLM Styled", 'Feedback'] = "Points"
exp2.loc[exp2['Condition'] == "IBL+LLM Selection Points Feedback", 'Feedback'] = "Points"
exp2.loc[exp2['Condition'] == "IBL+LLM Selection IBL+LLM Feedback", 'Feedback'] = "IBL+LLM"
exp2.loc[exp2['Condition'] == "Random Selection IBL+LLM Feedbac", 'Feedback'] = "IBL+LLM"
exp2.loc[exp2['Condition'] == "IBL Selection LLM Feedback", 'Feedback'] = "LLM"

mixed_anova = pg.mixed_anova(data=exp2, dv='Correct', between='Selection',  within='PhaseTrial', subject='UserId')
print(mixed_anova)

assert(False)
"""
\begin{tabular}{llrrrrrrrrrlrr}
\toprule
 & Source & SS & DF1 & DF2 & MS & F & p-unc & p-GG-corr & np2 & eps & sphericity & W-spher & p-spher \\
\midrule
0 & Selection & 4.794156 & 3 & 185 & 1.598052 & 3.057696 & 0.029599 & NaN & 0.047242 & NaN & NaN & NaN & NaN \\
1 & PhaseTrial & 10.418519 & 39 & 7215 & 0.267142 & 1.678999 & 0.005145 & 0.010573 & 0.008994 & 0.802754 & False & 0.006234 & 0.004730 \\
2 & Interaction & 20.970047 & 117 & 7215 & 0.179231 & 1.126478 & 0.167683 & NaN & 0.017940 & NaN & NaN & NaN & NaN \\
\bottomrule
\end{tabular}
"""

mixed_anova = pg.mixed_anova(data=exp2, dv='Correct', between='Feedback',  within='PhaseTrial', subject='UserId')
#print(mixed_anova)

mixed_anova = pg.mixed_anova(data=exp2, dv='Correct', between='Condition',  within='PhaseTrial', subject='UserId')
#print(mixed_anova)
"""
\begin{tabular}{lrrrrrrrrrlrr}
    \toprule
    Source & SS & DF1 & DF2 & MS & F & p-unc & p-GG-corr & np2 & eps & sphericity & W-spher & p-spher \\
    \midrule
    Condition & 2.560246 & 3 & 204 & 0.853415 & 0.829578 & 0.478967 & NaN & 0.012053 & NaN & NaN & NaN & NaN \\
    PhaseTrial & 5.772957 & 39 & 7956 & 0.148025 & 0.972538 & 0.518704 & 0.512240 & 0.004745 & 0.817006 & False & 0.010281 & 0.004749 \\
    Interaction & 18.914661 & 117 & 7956 & 0.161664 & 1.062150 & 0.306874 & NaN & 0.015380 & NaN & NaN & NaN & NaN \\
    \bottomrule
\end{tabular}
"""