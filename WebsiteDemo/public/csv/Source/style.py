import os 
import pandas as pd 
import numpy as np 
import openai
from tqdm import tqdm
import itertools
import time 
from bs4 import BeautifulSoup
from io import StringIO
from html.parser import HTMLParser
from tqdm import tqdm 

humanWrittenPlaintextEmails = pd.read_csv("./embedding_data.csv")
humanWrittenPlaintextEmails = humanWrittenPlaintextEmails.loc[:, ~humanWrittenPlaintextEmails.columns.str.contains('^Unnamed')]

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT_FRANCE") 
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY_FRANCE")

emailColumns = ["BaseEmailID", "Author", "Style", "Type", "Sender Style", "Sender", "Sender Mismatch", "Request Credentials", "Subject Suspicious", "Urgent", "Offer", "Link Mismatch", "Prompt", "Body", "Representation"]
emails = pd.DataFrame([], columns=emailColumns)

gpt_engine = "GPT4"

SenderName = ['Powerball','Paypal','Chase Bank','GoFundMe','Amazon','Microsoft','Chase Bank','Fedex','Indiana University','Walmart','Microsoft','Microsoft','Gmail','Paypal','Paypal','Amazon','JP Morgan Chase','USAA Bank','JP Morgan Chase','IRS','IRS','Amazon','Amazon','Chase Bank','Chase Bank','Ebay','IRS','USAA Bank','USA White House','USAA Bank','USAA Bank','Target','Paypal','Amazon','Walmart','K9 Security','Walmart','Amazon','Indiana University','A Personal Email','Help The Homeless','Paypal','A Personal Email','Indiana University','A Personal Email','Work From Home','Chase Bank','Chase Bank','A Personal Email','A Personal Email','A Personal Email','Chase Bank','Amazon','Amazon','Paypal','Amazon','Paypal','A Personal Email','Amazon','A Personal Email','Gmail','USA Insurance','Shopper','USAA Bank','A Personal Email','Huntington National Park','Verizon Wireless','JP Morgan Chase','Paypal','Google Docs','IRS','IRS','Reddit','LSU','IRS','A Personal Email','A Personal Email','Amazon','JP Morgan Chase','JP Morgan Chase','USAA Bank','JP Morgan Chase','JP Morgan Chase','Netflix','Indiana University','Indiana University','San Antonio Lottery','Amazon','Crowdsourced','Showcase','A Personal Email','Visa','Groupon','Indiana University','Amazon','Walmart','Amazon','UAM University','Indiana University','A Personal Email','Chase Bank','JP Morgan Chase','Chase Bank','Walmart','Chase Bank','IRS','Walmart','IRS','Walmart','Chase Bank','Microsoft','JP Morgan Chase','Chase Bank','A Personal Email','Paypal','Spotify','USAA Bank','Amazon','Paypal','Kaspersky Internet Security','Paypal','Chase Bank','Amazon Web Services','Google','A Personal Email','Jane Allen University Sweepstakes','Chase Bank','A Personal Email','A Personal Email','IRS','IRS','Paypal','Paypal','Amazon','Amazon','Chase Bank','Amazon','Amazon','Chase Bank','JP Morgan Chase Bank','JP Morgan Chase Bank','Amazon','Amazon','Amazon','A Personal Email','IRS','Indiana University','A Checking Company','Chase Bank','Blizzard','Blizzard','A Personal Email','Freedom Bank','Paypal','Class Mate Finder','Indiana University','Bank of America','Indiana','JP Morgan Chase','ABC Cellphone Company','Direct Loans','Chase Bank','USAA Bank','Walmart','Samsung','Indiana University','Indiana University','IRS','IRS','Amazon','Walmart','Walmart','Amazon','A Personal Email','Microsoft','Microsoft','Microsoft','A Personal Email','Gmail','A Personal Email','Chase Bank','Walmart','Chase Bank','A Personal Email','A Personal Email','Microsoft','Microsoft','A Personal Email','Thomas Cook Travels','Remitly','Remitly','Adobe','Adobe','Adobe','Amazon','Chiiz Photography Magazine','Google Scholar','UPS','UPS','UPS','Department of Homeland Security','A Personal Email','A Personal Email','United Airlines','A Personal Email','Piazza','Appolo University','Webster Towers','IEEE','Whatsapp','Eddie Bauer','CVS','Carnegie Mellon','A Research Conference','LinkedIn','Nike','Attack IQ','Twitter','Bigstock','Jobs.com','Google','Adorama','Paris College of Art','PNC Bank','A Research Conference','Chase Bank','TriActive','A Personal Email','Global Infrastructure','The Evans Group','Pop Money','A Personal Email','Carnegie Mellon University','Dropbox','CVS','Accenture Associate Recruitmen','Snapchat','Secarmour Security','Yahoo','Aetna','Dropbox','Credit Karma','Ebay','Bank of America','Bank of America','Paypal','Bank of America','Poloniex','Bank of America','Paypal','Coinbase','Paypal','Paypal','Apple','Amazon','Amazon','Bank of America','Our Revolution','Our Revolution','Nvidia','Nvidia','Mozilla','Bank of America','Bank of America','AirBnB','Google','Coinbase','Hedspace','Bank of America','2K Sports','Adobe','Digital Ocean','Bank of America','Bank of America','CVS','Arizona University','Arizona University','Arizona University','Arizona University','Simple Bills','Arizona University','Bill Thompson AVG Technologies','McAfee','Norton','A Personal Email','Apple','Bill Thompson AVG Technologies','McAfee','Arizona University','A Personal Email','Inside Insights','A Personal Email','A Personal Email','A Personal Email','Workmail','Arizona University','Change','Tuscon Libraries','A Personal Email','DMV','Arizona University','Pet Justice','Change','Change','Change','Change','Change','Change','Christians for Good','A Personal Email','A Personal Email','Facebook','Tuscon Libraries','A Personal Email','A Personal Email','Tully Elementary','A Personal Email','Tuscon Cooks','Tuscon Meets','A Personal Email','Tuscon Drinks','Amazon','Facebook','DFW Pharmacy','American Heart Association','DFW Pharmacy','Greyers Health Insurance','Amazon University','Northwest Allied Physicians','Georgia Research','Georgia Research','Georgia Health','Everyday Fitness','Everyday Fitness','Pure Protein','Eat Right','Globe Fitness','Body Smart Gym','Parking and Transportation','Legal Actions','Pariental Law','Parking and Transportation','Parking Services','Credit Debit Fraud Law','Law and Itentity Theft','Parental Law','Parental Law','Legal Law Firm','Kersey Law','Jury Duty','Jury Duty','Arizona University','Everyday Fitness','Weilys Texbooks','Dressbarn','Loyalty Bank','Loyalty Bank','American Breast Cancer Association','Arizona University','American Sky','Everyday Fitness','Chestnut','Shoe Warehouse','Shoe Warehouse','Marcys Wardrobe']

def get_response(prompt, max_tokens):
    try:
        response = openai.ChatCompletion.create(
            engine=gpt_engine, # engine = "deployment_name".
            messages=[
                {"role": "system", "content": "You are an HTML code writer that generates template emails that are used in educational examples to teach people how to identify whether an email has certain features."},
                {"role": "user", "content": prompt},
                ],
                temperature=1,# default is 1
                top_p=1,        # default is 1
                max_tokens=max_tokens
            )

        return response
    
    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return get_response(prompt, max_tokens) 

    except openai.error.APIError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return get_response(prompt, max_tokens) 

    except openai.error.ServiceUnavailableError as e:
        retry_time = 10  # Adjust the retry time as needed
        print(f"Service is unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return get_response(prompt, max_tokens) 

    except openai.error.Timeout as e:
        retry_time = 10  # Adjust the retry time as needed
        print(f"Request timed out: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return get_response(prompt, max_tokens) 

    except OSError as e:
        if isinstance(e, tuple) and len(e) == 2 and isinstance(e[1], OSError):
            retry_time = 10  # Adjust the retry time as needed
            print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return get_response(prompt, max_tokens) 
        else:
            retry_time = 10  # Adjust the retry time as needed
            print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            raise e

"""
['EmailID', 'Type', 'Company', 'Sender', 'Subject', 'Body', 'Link', 'LinkText',
       'sender_mismatch', 'requests_credentials', 'subject_suspicious',
       'urgent', 'offer', 'link_mismatch', 'Embeddings']
"""
#for index, row in humanWrittenPlaintextEmails.iterrows():    
for index, row in tqdm(humanWrittenPlaintextEmails.iterrows(), total=humanWrittenPlaintextEmails.shape[0]):
    if(index < 100): continue 
    sender_mismatch = row['sender_mismatch']
    subject_suspicious = row['subject_suspicious']
    requests_credentials = row['requests_credentials']
    link_mismatch = row['link_mismatch']
    urgent = row['urgent']
    offer = row['offer']

    # "EmailID", "Representation", "Body", "Author", "Style", "Prompt", "Type", "Sender Style", "Sender", , "Sender Mismatch", "Request Credentials", "Subject Suspicious", "Urgent", "Offer", "Link Mismatch"

    d = pd.DataFrame([{"EmailID": row['EmailID'], "Author": "Human", "Style":"Plaintext", "Type":row['Type'], "Sender Style":SenderName[index], "Sender":row['Sender'], "Sender Mismatch":row['sender_mismatch'], "Request Credentials":row['requests_credentials'], "Subject Suspicious":row['subject_suspicious'], "Urgent":row['urgent'], "Offer":row['offer'], "Link Mismatch":row['link_mismatch'], "Prompt":"None", "Body":row['Body'], "Representation": row['Embeddings'] }], columns=emailColumns)

    emails = pd.concat([emails, d])

    sender_mismatchString = "does" if sender_mismatch else "does not"
    requests_credentialsString = "does" if requests_credentials else "does not"
    subject_suspiciousString = "does" if subject_suspicious else "does not"
    urgentString = "does" if urgent else "does not"
    offerString = "does" if offer else "does not"
    link_mismatchString = "does" if link_mismatch else "does not"

    companyString = ' appears to be sent from ' + SenderName[index]

    # Your job is to generate email templates for educational purposes teaching people programming and online safety. 
    prompt = "Generate HTML code for an email with a header and banner and a logo where appropriate. Use a professional style that" + companyString + ". Ensure that the email " + sender_mismatchString + " have a sender mismatch." + " Ensure that the email " + requests_credentialsString + " request credentials." + " Ensure that the email " + subject_suspiciousString + " have a suspicious subject." + " Ensure that the email " + urgentString + " use urgent language." +  " Ensure that the email " + offerString + " make an offer. " +   " Ensure that the email " + link_mismatchString + " have a mismatched link. "

    response = get_response(prompt, 1000)

    gptEmailBody = response['choices'][0]['message']['content']
    
    if("```html" in gptEmailBody and "```" in gptEmailBody): 
        gptEmailBody = gptEmailBody.split("```html")[1]
        gptEmailBody = gptEmailBody.split("```")[0]
    
    if("<html>" in gptEmailBody and "</html>" in gptEmailBody):
        gptEmailBody = gptEmailBody.split("<html>")[1]
        gptEmailBody = gptEmailBody.split("</html>")[0]
    
    #print(gptEmailBody)

    representation = openai.Embedding.create(
        engine="ADA-002",
    input=[gptEmailBody])

    representation = representation["data"][0]["embedding"]

    d = pd.DataFrame([{"EmailID": row['EmailID'], "Author": "GPT", "Style":"GPT", "Type":row['Type'], "Sender Style":SenderName[index], "Sender":row['Sender'], "Sender Mismatch":row['sender_mismatch'], "Request Credentials":row['requests_credentials'], "Subject Suspicious":row['subject_suspicious'], "Urgent":row['urgent'], "Offer":row['offer'], "Link Mismatch":row['link_mismatch'], "Prompt":prompt, "Body":gptEmailBody, "Representation": representation,}], columns=emailColumns)

    emails = pd.concat([emails, d])

    plaintext  = BeautifulSoup(gptEmailBody, "lxml").text

    representation = openai.Embedding.create(
        engine="ADA-002",
    input=[plaintext])

    representation = representation["data"][0]["embedding"]

    d = pd.DataFrame([{"EmailID": row['EmailID'], "Author": "GPT", "Style":"Plaintext", "Type":row['Type'], "Sender Style":SenderName[index], "Sender":row['Sender'], "Sender Mismatch":row['sender_mismatch'], "Request Credentials":row['requests_credentials'], "Subject Suspicious":row['subject_suspicious'], "Urgent":row['urgent'], "Offer":row['offer'], "Link Mismatch":row['link_mismatch'], "Prompt":prompt, "Body":plaintext, "Representation": representation}], columns=emailColumns)

    emails = pd.concat([emails, d])

    #print(prompt)

    prompt = "Generate HTML code for an email with a header and banner and a logo where appropriate. Use a professional style that" + companyString + " Ensure that the email has the following text in the main body of the email " + row["Body"]

    response = get_response(prompt, 1000)

    gptEmailBody = response['choices'][0]['message']['content']
    
    if("```html" in gptEmailBody and "```" in gptEmailBody): 
        gptEmailBody = gptEmailBody.split("```html")[1]
        gptEmailBody = gptEmailBody.split("```")[0]
    
    if("<html>" in gptEmailBody and "</html>" in gptEmailBody):
        gptEmailBody = gptEmailBody.split("<html>")[1]
        gptEmailBody = gptEmailBody.split("</html>")[0]
    
    representation = openai.Embedding.create(
        engine="ADA-002",
    input=[gptEmailBody])

    representation = representation["data"][0]["embedding"]

    d = pd.DataFrame([{"EmailID": row['EmailID'], "Author":"Human", "Style":"GPT", "Type":row['Type'], "Sender Style":SenderName[index], "Sender":row['Sender'],  "Sender Mismatch":row['sender_mismatch'], "Request Credentials":row['requests_credentials'], "Subject Suspicious":row['subject_suspicious'], "Urgent":row['urgent'], "Offer":row['offer'], "Link Mismatch":row['link_mismatch'], "Prompt":prompt, "Body":gptEmailBody, "Representation": representation}], columns=emailColumns)

    emails = pd.concat([emails, d])

    emails.to_csv("./emails_temp.csv")

emails.to_csv("./emails.csv")

