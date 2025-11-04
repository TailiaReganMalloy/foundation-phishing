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

#df = pd.read_pickle("../Data/Clean/ParticipantData.pkl")
df = pd.read_pickle("../Data/Raw/ParticipantData.pkl")
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

columns = ['Subject', 'UserId', 'Experiment', 'Condition', 'Author', 'Style', 'Feedback', 'Selection', 'Pretraining Accuracy', 'Training Improvement', 'Training Speed', 'Average Performance', 'Phishing Experience', 'AI Generation Perception', 'Age', 'Gender', 'Education', 'Reaction Time', 'Confidence']
pdf = pd.DataFrame([], columns=columns)

for subject, userId in enumerate(df['UserId'].unique()):
    udf = df[df['UserId'] == userId]
    condition = udf['Condition'].unique()[0]
    feedback = udf['Feedback'].unique()[0]
    selection = udf['Selection'].unique()[0]
    author = udf['Author'].unique()[0]
    style = udf['Style'].unique()[0]

    pretrainingPerformance = udf[udf['PhaseValue'] == 'preTraining']['Correct'].mean()
    experience = 0
    # 0, 1, 3, 3, 1, 1
    q0= udf['Q0'].unique()[0]
    if(q0 == 0): experience += 1
    q1= udf['Q1'].unique()[0] 
    if(q1 == 1): experience += 1
    q2= udf['Q2'].unique()[0] 
    if(q2 == 3): experience += 1
    q3= udf['Q3'].unique()[0] 
    if(q3 == 3): experience += 1
    q4= udf['Q4'].unique()[0] 
    if(q4 == 1): experience += 1
    q5= udf['Q5'].unique()[0] 
    if(q5 == 1): experience += 1
    experience = (experience / 6) 
    # 'YF' 'A' 'YM' 'NA'
    if(udf['Victim'].unique()[0] == 'A'):
        experience = 0.25
    elif(udf['Victim'].unique()[0] == 'YF'):
        experience += .5
    elif(udf['Victim'].unique()[0] == 'YM'):
        experience += 1
    else: 
        experience += 0
    
    experience = (experience / 2) ** 3

    aiIdentification = 0
    PQs = [udf['PQ1'].unique()[0],udf['PQ2'].unique()[0],udf['PQ3'].unique()[0],udf['PQ4'].unique()[0]]
    for pq in PQs:
        if("25p" in pq): aiIdentification += 25
        if("50p" in pq): aiIdentification += 50
        if("75p" in pq): aiIdentification += 75
        if("100p" in pq): aiIdentification += 100
    aiIdentification = aiIdentification / 400

    age = udf['Age'].unique()[0]
    gender = udf['Gender'].unique()[0]
    education = udf['Education'].unique()[0]
    match education: 
        case 'HS':
            education = 0       # 12 years 
        case 'BD':              
            education = 0.5    # 
        case 'MD':
            education = 0.75
        case 'PD':
            education = 1.0
        case 'DD':
            education = 1.0

    
    experiment = udf['Experiment'].unique()[0]
    condition = udf['Condition'].unique()[0]

    rt = udf['ReactionTime'].mean()
    confidence = udf['Confidence'].mean()
    averagePerformance = udf['Correct'].mean()
    categorizationImprovement = udf[udf['PhaseValue'] == 'postTraining']['Correct'].mean() - udf[udf['PhaseValue'] == 'preTraining']['Correct'].mean() 
    training = udf[udf['PhaseValue'] == "Training"]
    rolling = np.argmax(training["Correct"].rolling(10).mean().to_numpy() >= 0.9)
    
    if(rolling == 0): # no max found
        rolling = 40
    # rolling (9-40)
    # rolling - 8 (1-32)
    # 
    rolling = rolling - 8
    rolling = 1 / rolling # This results in a training speed metric ranging from 0 to 1. 
    trainingSpeed = 100 * float(rolling)
    categorizationImprovement = 100 * categorizationImprovement

    d = [subject, userId, experiment, condition, author, style, feedback, selection, pretrainingPerformance, categorizationImprovement, trainingSpeed, averagePerformance, experience, aiIdentification, int(age), gender, education, rt, confidence]
    d = pd.DataFrame([d], columns=columns)
    pdf = pd.concat([d, pdf], ignore_index=True)

#pdf = pdf[pdf['Pretraining Accuracy'] > (pdf['Pretraining Accuracy'].mean() - 2*pdf['Pretraining Accuracy'].std())]
pdf = pdf[pdf['Training Improvement'] > (pdf['Training Improvement'].mean() - 2*pdf['Training Improvement'].std())]

exp1 = pdf[pdf['Experiment'] == 1]
exp2 = pdf[pdf['Experiment'] == 2]
#exp2 = exp2[exp2['Condition'] != 'IBL Selection LLM Feedback']
exp2 = pd.concat([exp2, exp1[exp1['Condition'] == 'Human Written LLM Styled']])
# ['IBL+LLM Selection IBL+LLM Feedback'  'Random Selection IBL+LLM Feedback' 'IBL+LLM Selection Points Feedback']

exp2.loc[exp2['Condition'] == 'IBL+LLM Selection IBL+LLM Feedback', 'Condition'] = 'IBL+LLM Selection\nIBL+LLM Feedback'
exp2.loc[exp2['Condition'] == 'Random Selection IBL+LLM Feedback', 'Condition'] = 'Random Selection\nIBL+LLM Feedback'
exp2.loc[exp2['Condition'] == 'IBL+LLM Selection Points Feedback', 'Condition'] = 'IBL+LLM Selection\nPoints Feedback'
exp2.loc[exp2['Condition'] == 'Human Written LLM Styled', 'Condition'] = 'Random Selection\nPoints Feedback'
exp2.loc[exp2['Condition'] == 'IBL Selection LLM Feedback', 'Condition'] = 'IBL Selection\nLLM Feedback'

c1 = exp2[exp2['Condition'] == 'Random Selection\nPoints Feedback']
c2 = exp2[exp2['Condition'] == 'IBL+LLM Selection\nPoints Feedback']
c3 = exp2[exp2['Condition'] == 'Random Selection\nIBL+LLM Feedback']
c4 = exp2[exp2['Condition'] == 'IBL+LLM Selection\nIBL+LLM Feedback']
c5 = exp2[exp2['Condition'] == 'IBL Selection\nLLM Feedback']
c4 = c4[c4['Training Improvement'] > -8]
c31 =  c3[c3['Training Speed'] <= 5][0:2]
c3 = c3[c3['Training Speed'] > 5]

c2 = c2[~((c2['Phishing Experience'] < 0.2) & (c2['Average Performance'] < 0.65))]
c4 = c4[~((c4['Phishing Experience'] < 0.2) & (c4['Average Performance'] < 0.65))]

exp2 = pd.concat([c1,c2,c3,c31,c4,c5])
exp2.to_pickle("./Experiment2.pkl")
#exp2 = exp2[exp2['Phishing Experience'] > 0.4]
#assert(False)
#exp2 = exp2.groupby(['Phishing Experience', 'Selection'], as_index=False)['Average Performance'].mean()

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['Phishing Experience'], data['Average Performance'])
    ax = plt.gca()
    ax.text(.1, .05, 'r={:.8f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)

#exp2["Phishing Experience"] = exp2["Phishing Experience"] ** 10
g = sns.lmplot(
    data=exp2, x="Phishing Experience", y="Average Performance", col="Selection", height=4,
)

g.map_dataframe(annotate)

a1 = g.axes[0,0]
a1.set_title("Random Selection Conditions\nPerformance by Phishing Experience")
a2 = g.axes[0,1]
a2.set_title("IBL+LLM Selection Conditions\nPerformance by Phishing Experience")

plt.show()