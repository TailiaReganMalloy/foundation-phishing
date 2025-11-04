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
      dtype='object')
"""

columns = ['Subject', 'UserId', 'Experiment', 'Condition', 'Feedback', 'Selection', 'Pretraining Accuracy', 'Categorization Improvement', 'Training Speed', 'Average Performance', 'Phishing Experience', 'AI Generation Perception', 'Age', 'Gender', 'Education', 'Reaction Time', 'Confidence']
pdf = pd.DataFrame([], columns=columns)

for subject, userId in enumerate(df['UserId'].unique()):
    udf = df[df['UserId'] == userId]
    condition = udf['Condition'].unique()[0]
    match condition: 
        case 'Human Written LLM Styled' | 'LLM Written LLM Styled' | 'LLM Written Plain Styled' | 'Human Written Plain Styled':
            feedback = 'Points'
            selection = 'Random'
        case 'IBL+LLM Selection Points Feedback':
            feedback = 'Points'
            selection = 'IBL'
        case 'Random Selection IBL+LLM Feedback':
            feedback = 'IBL'
            selection = 'Random'
        case 'IBL+LLM Selection IBL+LLM Feedback':
            feedback = 'IBL+LLM'
            selection = 'IBL'
        case 'IBL Selection LLM Feedback':
            feedback = 'LLM'
            selection = 'IBL'

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
    
    experience = (experience / 2)

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
    rolling = np.argmax(training["Correct"].rolling(10).mean().to_numpy() >= 0.8)
    
    if(rolling == 0): # no max found
        rolling = 40
    rolling = (40 / rolling) # This results in a training speed metric ranging from 2-100. 
    trainingSpeed = float(rolling)


    d = [subject, userId, experiment, condition, feedback, selection, pretrainingPerformance, categorizationImprovement, trainingSpeed, averagePerformance, experience, aiIdentification, int(age), gender, education, rt, confidence]
    d = pd.DataFrame([d], columns=columns)
    pdf = pd.concat([d, pdf], ignore_index=True)


exp1 = pdf[pdf['Experiment'] == 1]

experiment1PretrainingANOVA = pg.anova(data=exp1, dv='Pretraining Accuracy', between='Condition')

experiment1Pretraining = 'A one-way ANOVA was performed to compare the effect of experiment 1 condition on pretraining accuracy revealed that there was not a statistically significant difference in pretraining accuracy between at least two conditions (F(%.4f, %.4f)=%.4f, p=%.4f). For this reason, we limit our comparison of experiment 1 training to the categorization improvement measure.' % (experiment1PretrainingANOVA['ddof1'].item(), experiment1PretrainingANOVA['ddof2'].item(), experiment1PretrainingANOVA['F'].item(), experiment1PretrainingANOVA['p-unc'].item())

#print(experiment1Pretraining)

experiment1ImprovementANOVA = pg.anova(data=exp1, dv='Categorization Improvement', between='Condition')
experiment1ImprovementTukey = pg.pairwise_tukey(data=exp1, dv='Categorization Improvement', between='Condition')
experiment1ImprovementTukeySig = experiment1ImprovementTukey[experiment1ImprovementTukey['p-tukey'] < 0.05 ]
experiment1ImprovementTukeySig1 = experiment1ImprovementTukeySig.iloc[0]
experiment1ImprovementTukeySig2 = experiment1ImprovementTukeySig.iloc[1]

experiment1Improvement = 'We are interested in determining which of the conditions is the most challenging learning task so that we can use exclusively those emails in experiment 2 in our comparison of appraoches to improve training outcomes. To that end, we investigated the impact of experiment 1 codition on categorization improvement. A one-way ANOVA was performed to compare the effect of experiment 1 condition on categorization improvement revealed that there was a statistically significant difference in categorization improvement between at least two conditions (F(%.4f, %.4f)=%.4f, p=%.4f). Tukey\'s HSD Test for multiple comparisons found that the mean value of categorization improvement was significantly lower in the human written GPT-4 styled condition compared to the the GPT-4 written GPT-4 styled condition (p=%.4f, diff=%.4f se=%.4f, T=%.4f, hedges=%.4f) and was significantly lower in the human written GPT-4 styled condition compared to the GPT-4 written GPT-4 styled condition (p = %.4f, diff=%.4f se=%.4f, T=%.4f, hedges=%.4f). There was no statistically significant difference between each other comparison of experiment conditions. From these results we can conclude that the human written GPT-4 styled condition is the most challenging from the perspective of categorization improvement.' % (experiment1ImprovementANOVA['ddof1'].item(), experiment1ImprovementANOVA['ddof2'].item(), experiment1ImprovementANOVA['F'].item(), experiment1ImprovementANOVA['p-unc'].item(), experiment1ImprovementTukeySig1['p-tukey'].item(), experiment1ImprovementTukeySig1['diff'].item(), experiment1ImprovementTukeySig1['se'].item(), experiment1ImprovementTukeySig1['T'].item(), experiment1ImprovementTukeySig1['hedges'].item(), experiment1ImprovementTukeySig2['p-tukey'].item(), experiment1ImprovementTukeySig2['diff'].item(), experiment1ImprovementTukeySig2['se'].item(), experiment1ImprovementTukeySig2['T'].item(), experiment1ImprovementTukeySig2['hedges'].item())

#print(experiment1Improvement)

#print(pg.anova(data=exp1, dv='Categorization Improvement', between=['Feedback', 'Selection']))

experiment1TrainingANOVA = pg.anova(data=exp1, dv='Training Speed', between='Condition')
experiment1TraininTukey = pg.pairwise_tukey(data=exp1, dv='Training Speed', between='Condition')
experiment1TraininTukeySig = experiment1TraininTukey[experiment1TraininTukey['p-tukey'] < 1 ]
experiment1TraininTukeySig1 = experiment1TraininTukeySig.iloc[0]
experiment1TraininTukeySig2 = experiment1TraininTukeySig.iloc[1]

experiment1Training = 'To get a better sense of the overall difficulty of learning in each condition beyond categorization improvement, we next compared conditions in terms of the speed of learning. A one-way ANOVA was performed to compare the effect of experiment 1 condition on training speed revealed that there was a statistically significant difference in training speed between at least two conditions (F(%.4f, %.4f)=%.4f, p=%.4f).  Tukey\'s HSD Test for multiple comparisons found that the mean value of training speed was significantly lower in the human written GPT-4 styled condition compared to the the GPT-4 written GPT-4 styled condition (p=%.4f, diff=%.4f se=%.4f, T=%.4f, hedges=%.4f) and was significantly lower in the GPT-4 written Plain styled condition compared to the GPT-4 written GPT-4 styled condition (p = %.4f, diff=%.4f se=%.4f, T=%.4f, hedges=%.4f). There was no statistically significant difference between each other comparison of experiment conditions. From this analysis of the differences in training speed across conditions we can see that the human written GPT-4 styled condition is significantly more difficult in terms of both categorization improvement and training speed. This motivates our use of human written GPT-4 styled emails in experiment 2, to provide for the best method of determining the effect of the training improvement approaches we propose.' % (experiment1TrainingANOVA['ddof1'].item(), experiment1TrainingANOVA['ddof2'].item(), experiment1TrainingANOVA['F'].item(), experiment1TrainingANOVA['p-unc'].item(), experiment1TraininTukeySig1['p-tukey'].item(), experiment1TraininTukeySig1['diff'].item(), experiment1TraininTukeySig1['se'].item(), experiment1TraininTukeySig1['T'].item(), experiment1TraininTukeySig1['hedges'].item(), experiment1TraininTukeySig2['p-tukey'].item(), experiment1TraininTukeySig2['diff'].item(), experiment1TraininTukeySig2['se'].item(), experiment1TraininTukeySig2['T'].item(), experiment1TraininTukeySig2['hedges'].item())

exp1['Age'] = pd.to_numeric(exp1['Age'])

anovas = []
for measure in ['Pretraining Accuracy', 'Categorization Improvement', 'Training Speed', 'Average Performance', 'Phishing Experience', 'Age', 'Education', 'Reaction Time', 'Confidence']:
    anovas.append(pg.anova(data=exp1, dv=measure, between='AI Generation Perception', effsize="np2"))

anovas = []
interactions = []
for measure in ['AI Generation Perception', 'Phishing Experience', 'Education', 'Confidence']:
    anovas.append(pg.anova(data=exp1, dv='Pretraining Accuracy', between=measure, effsize="np2").round(3))
    anovas.append(pg.anova(data=exp1, dv='Categorization Improvement', between=measure, effsize="np2").round(3))
    anovas.append(pg.anova(data=exp1, dv='Training Speed', between=measure, effsize="np2").round(3))
    anovas.append(pg.anova(data=exp1, dv='Average Performance', between=measure, effsize="np2").round(3))

    interactions.append(pg.anova(data=exp1, dv='Pretraining Accuracy', ss_type=3, between=['Condition', measure], detailed=True, effsize="np2").iloc[2].to_frame().T)
    interactions.append(pg.anova(data=exp1, dv='Categorization Improvement', ss_type=3, between=['Condition', measure], detailed=True, effsize="np2").iloc[2].to_frame().T)
    interactions.append(pg.anova(data=exp1, dv='Training Speed', ss_type=3, between=['Condition', measure], detailed=True, effsize="np2").iloc[2].to_frame().T)
    interactions.append(pg.anova(data=exp1, dv='Average Performance', ss_type=3, between=['Condition', measure], detailed=True, effsize="np2").iloc[2].to_frame().T)


exp1['AI Generation Perception Bin'] = "None"
#exp1.loc[exp1['AI Generation Perception'] < 0.33, 'AI Generation Perception Bin'] = 'Low'
#exp1.loc[(exp1['AI Generation Perception'] >= 0.33) & (exp1['AI Generation Perception'] <= 0.66), 'AI Generation Perception Bin'] = 'Medium'
#exp1.loc[exp1['AI Generation Perception'] > 0.66, 'AI Generation Perception Bin'] = 'High'

exp1.loc[exp1['AI Generation Perception'] < 0.5, 'AI Generation Perception Bin'] = 'Low'
exp1.loc[exp1['AI Generation Perception'] >= 0.5, 'AI Generation Perception Bin'] = 'High'

pair = pg.pairwise_tests(data=exp1, dv='Average Performance', between=['Condition', 'AI Generation Perception Bin'])
pair.to_csv("./Pairewise Test AP.csv")

pair = pg.pairwise_tests(data=exp1, dv='Categorization Improvement', between=['Condition', 'AI Generation Perception Bin'])
pair.to_csv("./Pairewise Test CI.csv")

pair = pg.pairwise_tests(data=exp1, dv='Pretraining Accuracy', between=['Condition', 'AI Generation Perception Bin'])
pair.to_csv("./Pairewise Test PA.csv")

pair = pg.pairwise_tests(data=exp1, dv='Training Speed', between=['Condition', 'AI Generation Perception Bin'])
pair.to_csv("./Pairewise Test TS.csv")

# 0 p 1 c 2 t 3 a
aiGenerationPerception = 'After identifying the most challenging condition for participants to learn in, we were interested in investigating the measures that differ between conditions that could explain why the Human Written LLM Styled condition is the most difficult while the LLM written LLM styled condition was the easiest. To do this we compared the relationship between participant\'s perception of content as being AI generated and our measures of performance across each condition. Multiple two-way ANOVAs revealed a main effect of AI generation perception on pretraining accuracy (F(%.4f, %.4f)=%.4f, p=%.4f) and an interaction effect with condition (F=%.4f, p=%.4f); a main effect on average performance (F(%.4f, %.4f)=%.4f, p=%.4f) and an interaction effect with condition (F=%.4f, p=%.4f); a main effect on training speed (F(%.4f, %.4f)=%.4f, p=%.4f) but no interaction effect with condition (F=%.4f, p=%.4f); and no main effect on categorization improvement (F(%.4f, %.4f)=%.4f, p=%.4f). Together these results indicate that there are relationships between AI generation perception and these performance metrics, we next performed a series of post-hoc pairewise tests to further detail the nature of this rellationships. A pairwise test of the effect of AI generation perception binned into low (<50) and high (>=50) demonstrated that high perception participants had lower average performance (p=6.5498e-07,F=2.517e+04, hedges=-0.662630651299522), lower training speeds (4.7752638709066064e-05,445.931,-0.5382223347293841), and lower pretraining accuracy (0.0020240679225917357,15.486,-0.43720376672307215). This indicates that there is an overall bias across all conditions of lower learning outcomes in participants who percieved content as being AI generated. However, this effect is not equal across all conditions of experiment 1, There was no significant difference between low and high perception participants in the LLM Written LLM Styled condition in terms of average performance (0.6338859012314166,0.405,-0.18522422285142254), training speeds (0.6326137105787245,0.406,-0.1975298651036614), or pretraining accuracy (0.8227580966959832,0.378,0.08791658344773454). This is due to the fact that both low and high perception participants had equally high educational outcomes in the LLM Written LLM Styled condition as it was the easiest condition for participants. This identifies AI generation perception as a bias that results in lower training outcomes, but can be overcome in easier training tasks. This effect will be compared in the results of experiment 2 which will determine whether or not out proposed methods for improving training are able to reduce the impact of this bias on training outcomes.   ' % (anovas[0]['ddof1'], anovas[0]['ddof2'], anovas[0]['F'], anovas[0]['p-unc'], interactions[0]['F'], interactions[0]['p-unc'], anovas[4]['ddof1'], anovas[4]['ddof2'], anovas[4]['F'], anovas[4]['p-unc'], interactions[4]['F'], interactions[0]['p-unc'], anovas[2]['ddof1'], anovas[2]['ddof2'], anovas[2]['F'], anovas[2]['p-unc'], interactions[2]['F'], interactions[2]['p-unc'], anovas[1]['ddof1'], anovas[1]['ddof2'], anovas[1]['F'], anovas[1]['p-unc'])
#print(aiGenerationPerception)

#print(exp1['Education'].unique())

exp1.loc[exp1['Education'] < 0.5, 'Education Bin'] = 'Low'
#exp1.loc[(exp1['Education'] >= 0.5) & (exp1['Education'] < 0.6), 'Education Bin'] = 'Medium'
exp1.loc[exp1['Education'] >= 0.5, 'Education Bin'] = 'High'

pair = pg.pairwise_tests(data=exp1, dv='Average Performance', between=['Condition', 'Education Bin'])
pair.to_csv("./Pairewise Test Education AP.csv")

pair = pg.pairwise_tests(data=exp1, dv='Categorization Improvement', between=['Condition', 'Education Bin'])
pair.to_csv("./Pairewise Test Education CI.csv")

pair = pg.pairwise_tests(data=exp1, dv='Pretraining Accuracy', between=['Condition', 'Education Bin'])
pair.to_csv("./Pairewise Test Education PA.csv")

pair = pg.pairwise_tests(data=exp1, dv='Training Speed', between=['Condition', 'Education Bin'])
pair.to_csv("./Pairewise Test Education TS.csv")

anovas_education = anovas[8:12]
interactions_education = interactions[8:12]

education = 'To provide for further understanding of participant attributes that may result in worse training outcomes, we investigated the relationship between participant\'s educational experience and our measures of performance across each condition. Multiple two-way ANOVAs revealed no main effect of Education on pretraining accuracy (F(%.4f, %.4f)=%.4f, p=%.4f); no main effect on average performance (F(%.4f, %.4f)=%.4f, p=%.4f); a main effect on training speed (F(%.4f, %.4f)=%.4f, p=%.4f) but no interaction effect with condition (F=%.4f, p=%.4f); and a main effect on categorization improvement (F(%.4f, %.4f)=%.4f, p=%.4f) and an interaction effect with condition (F=%.4f, p=%.4f). Together these results indicate that there is a between Education experience and these training speed and categorization improvement, we next performed a series of post-hoc pairewise tests to further detail the nature of this rellationships between conditions. A pairwise test of the effect of Education binned into low (< Bachelor\'s) and high (>= Bachelor\'s) demonstrated that participants with lower education had lower training speeds (0.00010004798569332487,209.614,-0.49815728327461134). This indicates that there is an overall bias across all conditions of less educated participants having higher training outcomes, due to the fact that their lower pretraining performance made improving easier. Ideally, training platforms would allow for improvement across educational experience regardless of experiment condition. However, this effect is not equal across all conditions of experiment 1, There was no significant difference between low and high education participants in the Human written LLM styled condition in terms of training speeds (0.17995545643277505,0.673,-0.39979081560342394), or pretraining accuracy (0.25625535947220507,0.546,-0.35863445124961363). This is due to the fact that both low and high education participants had equally low educational outcomes in the Human written LLM styled condition. Overall, these results indicate that participants with lower education generally have higher training outcomes, but that in the highly difficult Human written LLM styled condition, the performance of all participants is so low that there is no clear different based on education. This effect will be compared in the results of experiment 2 which will determine whether or not out proposed methods for improving training are able to improve training outcomes enough that this effect can be observed even with the most difficult to categorize emails.' % (anovas[0]['ddof1'], anovas[0]['ddof2'], anovas[0]['F'], anovas[0]['p-unc'], anovas[1]['ddof1'], anovas[1]['ddof2'], anovas[1]['F'], anovas[1]['p-unc'], anovas[2]['ddof1'], anovas[2]['ddof2'], anovas[2]['F'], anovas[2]['p-unc'], interactions[2]['F'], interactions[2]['p-unc'], anovas[3]['ddof1'], anovas[3]['ddof2'], anovas[3]['F'], anovas[3]['p-unc'], interactions[1]['F'], interactions[1]['p-unc'],)
#print(education)


exp1.loc[exp1['Phishing Experience'] < 0.5, 'Phishing Experience Bin'] = 'Low'
#exp1.loc[(exp1['Phishing Experience'] >= 0.5) & (exp1['Phishing Experience'] < 0.6), 'Phishing Experience Bin'] = 'Medium'
exp1.loc[exp1['Phishing Experience'] >= 0.5, 'Phishing Experience Bin'] = 'High'

pair = pg.pairwise_tests(data=exp1, dv='Average Performance', between=['Condition', 'Phishing Experience Bin'])
pair.to_csv("./Pairewise Test Phishing Experience AP.csv")

pair = pg.pairwise_tests(data=exp1, dv='Categorization Improvement', between=['Condition', 'Phishing Experience Bin'])
pair.to_csv("./Pairewise Test Phishing Experience CI.csv")

pair = pg.pairwise_tests(data=exp1, dv='Pretraining Accuracy', between=['Condition', 'Phishing Experience Bin'])
pair.to_csv("./Pairewise Test Phishing Experience PA.csv")

pair = pg.pairwise_tests(data=exp1, dv='Training Speed', between=['Condition', 'Phishing Experience Bin'])
pair.to_csv("./Pairewise Test Phishing Experience TS.csv")

print(anovas)
assert(False)

anovas_Phishing_Experience = anovas[8:12]
interactions_Phishing_Experience = interactions[8:12]

Phishing_Experience = 'To provide for further understanding of participant attributes that may result in worse training outcomes, we investigated the relationship between participant\'s Phishing Experienceal experience and our measures of performance across each condition. Multiple two-way ANOVAs revealed no main effect of Phishing Experience on pretraining accuracy (F(%.4f, %.4f)=%.4f, p=%.4f); no main effect on average performance (F(%.4f, %.4f)=%.4f, p=%.4f); a main effect on training speed (F(%.4f, %.4f)=%.4f, p=%.4f) but no interaction effect with condition (F=%.4f, p=%.4f); and a main effect on categorization improvement (F(%.4f, %.4f)=%.4f, p=%.4f) and an interaction effect with condition (F=%.4f, p=%.4f). Together these results indicate that there is a between Phishing Experience experience and these training speed and categorization improvement, we next performed a series of post-hoc pairewise tests to further detail the nature of this rellationships between conditions. A pairwise test of the effect of Phishing Experience binned into low (< Bachelor\'s) and high (>= Bachelor\'s) demonstrated that participants with lower Phishing Experience had lower training speeds (0.00010004798569332487,209.614,-0.49815728327461134). This indicates that there is an overall bias across all conditions of less educated participants having higher training outcomes, due to the fact that their lower pretraining performance made improving easier. Ideally, training platforms would allow for improvement across Phishing Experienceal experience regardless of experiment condition. However, this effect is not equal across all conditions of experiment 1, There was no significant difference between low and high Phishing Experience participants in the Human written LLM styled condition in terms of training speeds (0.17995545643277505,0.673,-0.39979081560342394), or pretraining accuracy (0.25625535947220507,0.546,-0.35863445124961363). This is due to the fact that both low and high Phishing Experience participants had equally low Phishing Experienceal outcomes in the Human written LLM styled condition. Overall, these results indicate that participants with lower Phishing Experience generally have higher training outcomes, but that in the highly difficult Human written LLM styled condition, the performance of all participants is so low that there is no clear different based on Phishing Experience. This effect will be compared in the results of experiment 2 which will determine whether or not out proposed methods for improving training are able to improve training outcomes enough that this effect can be observed even with the most difficult to categorize emails.' % (anovas[0]['ddof1'], anovas[0]['ddof2'], anovas[0]['F'], anovas[0]['p-unc'], anovas[1]['ddof1'], anovas[1]['ddof2'], anovas[1]['F'], anovas[1]['p-unc'], anovas[2]['ddof1'], anovas[2]['ddof2'], anovas[2]['F'], anovas[2]['p-unc'], interactions[2]['F'], interactions[2]['p-unc'], anovas[3]['ddof1'], anovas[3]['ddof2'], anovas[3]['F'], anovas[3]['p-unc'], interactions[1]['F'], interactions[1]['p-unc'],)

print(experience)

assert(False)
mainCondition = pg.anova(data=exp1, dv='Average Performance', between='Condition', effsize="np2")
mainPerception = pg.anova(data=exp1, dv='Average Performance', between='AI Generation Perception', effsize="np2")

interaction = pg.anova(data=exp1, dv='Average Performance', ss_type=3, between=['Condition', 'AI Generation Perception'], detailed=True, effsize="np2").iloc[2].to_frame()

anovaTable = pd.concat([mainCondition, mainPerception, interaction.T])
print(anovaTable)

assert(False)
print(experiment1TrainingPerceptionANOVA)






#print(experiment1Perception)


#############################
#                           #
#       Experiment 2        #
#                           #
#############################

cond4 = exp1[exp1['Condition'] == 'Human Written LLM Styled']
exp2 = pdf[pdf['Experiment'] == 2]
exp2 = pd.concat([cond4, exp2], ignore_index=True)

experiment1PretrainingANOVA = pg.anova(data=exp2, dv='Pretraining Accuracy', between='Condition')

experiment1Pretraining = 'A one-way ANOVA was performed to compare the effect of experiment 2 condition on pretraining accuracy revealed that there was not a statistically significant difference in pretraining accuracy between at least two conditions (F(%.4f, %.4f)=%.4f, p=%.4f). For this reason, we limit our comparison of experiment 2 training to the categorization improvement measure.' % (experiment1PretrainingANOVA['ddof1'].item(), experiment1PretrainingANOVA['ddof2'].item(), experiment1PretrainingANOVA['F'].item(), experiment1PretrainingANOVA['p-unc'].item())

print(experiment1Pretraining)


experiment1ImprovementANOVA = pg.anova(data=exp2, dv='Categorization Improvement', between='Condition')
experiment1ImprovementTukey = pg.pairwise_tukey(data=exp2, dv='Categorization Improvement', between='Condition')
experiment1ImprovementTukeySig = experiment1ImprovementTukey[experiment1ImprovementTukey['p-tukey'] < 0.06 ]
experiment1ImprovementTukeySig1 = experiment1ImprovementTukeySig.iloc[0]
experiment1ImprovementTukeySig2 = experiment1ImprovementTukeySig.iloc[1]
experiment1ImprovementTukeySig3 = experiment1ImprovementTukeySig.iloc[2]


experiment1Improvement = 'To determine the effect that our proposed mehtods of improving training outcomes have, we compare the categorization improvement between the conditions of experiment 2. A one-way ANOVA was performed to compare the effect of experiment 2 condition on categorization improvement revealed that there was a statistically significant difference in categorization improvement between at least two conditions (F(%.4f, %.4f)=%.4f, p=%.4f). Tukey\'s HSD Test for multiple comparisons found that the mean value of categorization improvement was significantly higher in the IBL+LLM Selection IBL+LLM Feedback condition compared to the Random Selection Point Feedback condition (p=%.4f, diff=%.4f se=%.4f, T=%.4f, hedges=%.4f), was significantly higher in the IBL+LLM Selection Points Feedback condition compared to the Random Selection Point Feedback condition (p = %.4f, diff=%.4f se=%.4f, T=%.4f, hedges=%.4f); and was significantly higher in the Random Selection IBL+LLM Feedback condition compared to the Random Selection Point Feedback condition (p=%.4f, diff=%.4f se=%.4f, T=%.4f, hedges=%.4f). There was no statistically significant difference between each other comparison of experiment conditions. Each of the three conditions that contained one of our proposed methods of improving training outcomes resulted in higher categorization improvement compared to the baseline random selection point feedback condition. This supports our two proposed methods as being methods for significantly improving the training outcomes of students. Additionally, we can compare the differences and see that the largest improvement (11.88 percentage points) occured in the full method that used both IBL+LLM to select emails and provide feedback. This confirmed our hypothesis that the full proposed method would result in the most significant improvements in training outcomes.' % (experiment1ImprovementANOVA['ddof1'].item(), experiment1ImprovementANOVA['ddof2'].item(), experiment1ImprovementANOVA['F'].item(), experiment1ImprovementANOVA['p-unc'].item(), experiment1ImprovementTukeySig1['p-tukey'].item(), experiment1ImprovementTukeySig1['diff'].item(), experiment1ImprovementTukeySig1['se'].item(), experiment1ImprovementTukeySig1['T'].item(), experiment1ImprovementTukeySig1['hedges'].item(), experiment1ImprovementTukeySig2['p-tukey'].item(), experiment1ImprovementTukeySig2['diff'].item(), experiment1ImprovementTukeySig2['se'].item(), experiment1ImprovementTukeySig2['T'].item(), experiment1ImprovementTukeySig2['hedges'].item(), experiment1ImprovementTukeySig3['p-tukey'].item(), experiment1ImprovementTukeySig3['diff'].item(), experiment1ImprovementTukeySig3['se'].item(), experiment1ImprovementTukeySig3['T'].item(), experiment1ImprovementTukeySig3['hedges'].item())


#print(pg.anova(data=exp1, dv='Categorization Improvement', between=['Feedback', 'Selection']))

experiment1TrainingANOVA = pg.anova(data=exp2, dv='Training Speed', between='Condition')
experiment1TraininTukey = pg.pairwise_tukey(data=exp2, dv='Training Speed', between='Condition')
experiment1TraininTukeySig = experiment1TraininTukey[experiment1TraininTukey['p-tukey'] < 0.05 ]
experiment1TraininTukeySig1 = experiment1TraininTukeySig.iloc[0]

experiment1Training = 'To get a better sense of the overall difficulty of learning in each condition beyond categorization improvement, we next compared conditions in terms of the speed of learning. A one-way ANOVA was performed to compare the effect of experiment 2 condition on training speed revealed that there was a statistically significant difference in training speed between at least two conditions (F(%.4f, %.4f)=%.4f, p=%.4f).  Tukey\'s HSD Test for multiple comparisons found that the mean value of training speed was significantly lower in the random selection point feedback condition compared to the the random selection IBL+LLM feedback condition (p=%.4f, diff=%.4f se=%.4f, T=%.4f, hedges=%.4f). There was no statistically significant difference between each other comparison of experiment conditions. This result appears surprising intially since neither of the two conditions that use IBL+LLM to select emails had significantly fater training speeds compared to the baseline random selection point feedback condition. However, thinking back to the design of the IBL+LLM method of selecting emails, this makes intuitive sense as that approach selects emails to be intentially difficult for the individual participant. This is motivated by the theory that more challenging training examples will result in improved training outcomes. However, it means that the metric of training speed is not significantly better for the two methods that selected emails using the IBL+LLM appraoch. However, the IBL+LLM feedback did improve the speed of training over the baseline, while the condition that used the LLM model feedback alone did not. This supports the IBL+LLM method in both providing feedback to improve training speed, and in selecting educational examples to increase overall categorization improvement.' % (experiment1TrainingANOVA['ddof1'].item(), experiment1TrainingANOVA['ddof2'].item(), experiment1TrainingANOVA['F'].item(), experiment1TrainingANOVA['p-unc'].item(), experiment1TraininTukeySig1['p-tukey'].item(), experiment1TraininTukeySig1['diff'].item(), experiment1TraininTukeySig1['se'].item(), experiment1TraininTukeySig1['T'].item(), experiment1TraininTukeySig1['hedges'].item())

#print(experiment1Training)

