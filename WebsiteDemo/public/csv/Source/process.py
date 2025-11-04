import pandas as pd 
from reader import Reader
import csv

emails = pd.read_csv("./processedEmails.csv")
data = pd.read_csv("data.csv")


for index, row in emails.iterrows():
    if(row['Style'] == "Plaintext"):
        link = data[data["EmailID"] == row['BaseEmailID']]["Link"].item()
        linkText = data[data["EmailID"] == row['BaseEmailID']]["LinkText"].item()
        # class=""link"">none<
        newLink = '<a onclick=""javascript:return false;""  href=""' + link + '"" class=""link"">' + linkText + '</a>'
        if(linkText in row['Body']):
            newBody = row['Body'].replace(linkText, newLink)
        else:
            newBody = row['Body'] + "\n" + newLink 

        emails.at[index, 'Body'] = newBody

#emails["EmailID"] = range(0,1464)
emails.to_csv("./processedEmails.csv", index=False)

# 

"""path = "./emails.csv"
embedding_data = pd.read_csv("./embedding_data.csv")
# EmailID,Type,Sender,Subject,Body,Link,LinkText,sender_mismatch,requests_credentials,subject_suspicious,urgent,offer,link_mismatch,Embeddings
# open file in read mode
line = 0 

columns = ['BaseEmailID', 'Author', 'Style', 'Type', 'Sender Style', 'Sender', 'Sender Mismatch', 'Request Credentials', 'Subject Suspicious', 'Urgent', 'Offer', 'Link Mismatch', 'Prompt', 'Body', 'Representation']
df = pd.DataFrame([], columns=columns)

with open("./emails_100.csv", newline='', encoding='utf-8') as csvfile:    
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    BaseEmailID = 0
    
    for row_index, row in enumerate(reader):
        if(row_index == 0): continue
        
        body_match = embedding_data[embedding_data['Body'] == row[14]]
        if(len(body_match) == 1):
            BaseEmailID = body_match['EmailID'].item()
        
        
        d = {'BaseEmailID':BaseEmailID, 'Author':row[2], 'Style':row[3], 'Type':row[4], 'Sender Style':row[5], 'Sender':row[6], 'Sender Mismatch':row[7], 'Request Credentials':row[8], 'Subject Suspicious':row[9], 'Urgent':row[10], 'Offer':row[11], 'Link Mismatch':row[12], 'Prompt':row[13], 'Body':row[14], 'Representation':row[15]}
        if(len(d) != 15): assert(False)
        d = pd.DataFrame([d])
        df = pd.concat([df, d])

with open(path, newline='', encoding='utf-8') as csvfile:    
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    BaseEmailID = 0
    
    for row_index, row in enumerate(reader):
        if(row_index == 0): continue
        
        body_match = embedding_data[embedding_data['Body'] == row[14]]
        if(len(body_match) == 1):
            BaseEmailID = body_match['EmailID'].item()
        
        d = {'BaseEmailID':BaseEmailID, 'Author':row[2], 'Style':row[3], 'Type':row[4], 'Sender Style':row[5], 'Sender':row[6], 'Sender Mismatch':row[7], 'Request Credentials':row[8], 'Subject Suspicious':row[9], 'Urgent':row[10], 'Offer':row[11], 'Link Mismatch':row[12], 'Prompt':row[13], 'Body':row[14], 'Representation':row[15]}
        if(len(d) != 15): assert(False)
        d = pd.DataFrame([d])
        df = pd.concat([df, d])

print(len(df))

df.dropna(how='all', axis='columns')

df.to_csv("./processedEmails.csv")

data = pd.read_csv("./processedEmails.csv")
print("done")"""