import pandas as pd 

df = pd.read_csv("processedEmails.csv")

import pandas as pd 
from reader import Reader
import csv
import math 

emails = pd.read_csv("./processedEmails copy.csv")

emails.drop(['Prompt', 'Representation'], axis=1, inplace=True)

emails.to_csv("./processedEmails.csv", index=False)
