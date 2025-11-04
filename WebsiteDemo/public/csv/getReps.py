import pandas as pd 

df = pd.read_csv("processedEmailsReps.csv")
df['Representation'].to_csv("./representations.csv", index=False)

#emails.drop(['Prompt', 'Representation'], axis=1, inplace=True)

#emails.to_csv("./representations.csv", index=False)