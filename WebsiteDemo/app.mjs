import express from 'express'
const app = express();

import session from 'express-session'
import path from 'path'
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const dotenv = require('dotenv');

dotenv.config();

import fs from 'fs'
console.log(process.env.OPENAI_API_KEY)
console.log(process.env.OPENAI_BASE_URL)
process.setMaxListeners(0);

import {Server as HTTPServer} from 'http'
const http = HTTPServer(app)
import {Server, Socket} from 'socket.io';
const io = new Server(http)

import {parse} from 'csv-parse';
import { AzureOpenAI } from "openai";

// Human Written, GPT-4 Stylized
var Condition3Phishing = [103, 107, 115, 119, 127, 131, 135, 147, 143, 151, 163]
var Condition3Ham      = [759, 763, 771, 783, 787, 795, 1007, 1047, 1063, 1253, 1077]

// GPT-4 Written, GPT-4 Stylized
var Condition4Phishing = [101, 105, 113, 117, 125, 129, 133, 145, 141, 149, 161]
var Condition4Ham      = [757, 761, 769, 781, 785, 793, 1005, 1045, 1061, 1255, 1079]

const openAPIclient = new AzureOpenAI();

export {openAPIclient};

var messages=[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Who won the world series in 2020?"},
  {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
  {"role": "user", "content": "Where was it played?"}
]

openAPIclient.chat.completions.create({messages, model:'gpt-4o-mini', max_tokens: 150})
.then(result => {
    console.log(result.choices[0].message.content)
  }
).catch(err => {
  console.log(err)
});



import bodyParser from  'body-parser';
// Add bodyParser middleware for handling URL-encoded data
app.use(bodyParser.urlencoded({ extended: true }));
// Add bodyParser middleware for handling JSON data
app.use(bodyParser.json());

import * as url from 'url';
  const __filename = url.fileURLToPath(import.meta.url);
  const __dirname = url.fileURLToPath(new URL('.', import.meta.url));

// Serve static files from the 'public' folder
app.use('/public', express.static(path.join(__dirname, 'public')));
app.use('/html', express.static(path.join(__dirname, 'public/html')));
app.use('/css', express.static(path.join(__dirname, 'public/css')));
app.use('/images', express.static(path.join(__dirname, 'images')));

var emailData=[];
var skip = 0
fs.createReadStream(path.join(__dirname, 'public/csv/processedEmails.csv'))
  .pipe(parse({delimiter: ','}))
  .on('data', function(csvrow) {
    if(skip == 0){
      skip = 1
    }else{
      emailData.push(csvrow); 
    }
});

var representations=[];
fs.createReadStream(path.join(__dirname, 'public/csv/representations.csv'))
  .pipe(parse({delimiter: ','}))
  .on('data', function(csvrow) {
    representations.push(csvrow);
});



function dotProduct(vec1, vec2) {
  if (vec1.length !== vec2.length) {
      throw new Error('Vectors must have the same length');
  }

  let dot = 0;
  for (let i = 0; i < vec1.length; i++) {
      dot += vec1[i] * vec2[i];
  }
  return dot;
}

function magnitude(vec) {
  let mag = 0;
  for (let i = 0; i < vec.length; i++) {
      mag += vec[i] * vec[i];
  }
  return Math.sqrt(mag);
}

function cosineSimilarity(vec1, vec2) {
  if(vec1.length === 0 || vec2.length === 0){
    return 0 
  }
  const dot = dotProduct(vec1, vec2);
  const mag1 = magnitude(vec1);
  const mag2 = magnitude(vec2);
  return 1 - ((dot / (mag1 * mag2)) / 2);
}


function randomExcluded(list, min, max, excluded) {
  var n = Math.floor(Math.random() * (max-min) + min).toString();
  var val = list[n].toString()
  if (excluded.includes(val)){
    return randomExcluded(list, min, max, excluded)
  }else{
    return parseInt(val);
  }
}

function meanOfVectors(vectors) {
  if (vectors.length === 0) {
      return null; // Handle empty array case
  }

  const vectorLength = vectors[0].length;
  const meanVector = new Array(vectorLength).fill(0);

  vectors.forEach(vector => {
      if (vector.length !== vectorLength) {
          return 0;
          throw new Error('All vectors must have the same length');
      }

      vector.forEach((value, index) => {
          meanVector[index] += value;
      });
  });

  meanVector.forEach((value, index) => {
      meanVector[index] = value / vectors.length;
  });

  return meanVector;
}


app.use(session({
  secret: "None", // Change this to a secure secret key
  resave: true,
  saveUninitialized: true
}));

app.get('/random/:Condition/:Type', function(req, res){
  if(req.params.Condition == 1 && req.params.Type == "phishing"){
    var index = randomExcluded(Condition1Phishing, 0,188,[])
    res.send(emailData[index][14])
  }
});

app.get('/trial/:Trial', function(req, res){
  // ,BaseEmailID,Author,Style,Type,Sender Style,Sender,Sender Mismatch,Request Credentials,Subject Suspicious,Urgent,Offer,Link Mismatch,Prompt,Body,Representation
  if(req.params.Trial < emailData.length){
    res.send(emailData[req.params.Trial][14])
  }else{
    res.send("Try another trial")
  }
  
});

app.get('/reject/:MID', function(req, res){
  // ,MTurkId,Age,Gender,Education,Country,Victim,Consent,Q0,Q1,Q2,Q3,Q4,Q5,Userid,PQ0, PQ1, PQ2, PQ3, PQ4, PQ5, Rejected,
  var Mturkid = req.params.MID.replace(/[^a-zA-Z ]/g, "")
  var Moment = moment().format('yyyy-MM-dd:hh:mm:ss')
  res.sendFile('public/html/reject.html', {root: __dirname })
});

app.get('/postQuestionnaire', function(req, res){
  res.sendFile('public/html/postQuestionnaire.html', {root: __dirname })
});

app.get('/content', function(req, res){
  var EmailType = req.query.EmailType
  var Condition = parseInt(req.query.Condition)
  var UserId = req.query.UserId
  var PreviouslyObserved = req.query.PreviouslyObserved
  var PhaseValue = req.query.PhaseValue
  PreviouslyObserved = PreviouslyObserved.split(",")
  var Trial = req.query.Trial

  if(Condition === 1){
    if(EmailType == "ham"){
      var index = Condition3Ham[Trial]
      var email = emailData[index]
      res.send(email.join(','))
    }else{
      var index = Condition3Phishing[Trial]
      var email = emailData[index]
      res.send(email.join(','))
    }
  }else if(Condition === 2){
    if(EmailType == "ham"){
      var index = Condition4Ham[Trial]
      var email = emailData[index]
      res.send(email.join(','))
    }else{
      var index = Condition4Phishing[Trial]
      var email = emailData[index]
      res.send(email.join(','))
    }
  }else{
    res.send("Error fetching email")
  }
  

});  

app.get('/insert', function(req, res){
  res.send("success")
});

app.get('/questionnaire', function(req, res){
  var MTurkId = req.query.MTurkId
  var Age = req.query.Age
  var Gender = req.query.Gender
  var Education = req.query.Education
  var Country = req.query.Country
  var Victim = req.query.Victim
  var Chatbot = req.query.Chatbot
  var Consent = req.query.Consent
  var Q0 = req.query.Q0
  var Q1 = req.query.Q1
  var Q2 = req.query.Q2
  var Q3 = req.query.Q3
  var Q4 = req.query.Q4
  var Q5 = req.query.Q5
  var Userid = req.query.Userid
  var PQ0 = req.query.PQ0
  var PQ1 = req.query.PQ1
  var PQ2 = req.query.PQ2
  var PQ3 = req.query.PQ3
  var PQ4 = req.query.PQ4
  var PQ5 = req.query.PQ5
  var Rejected = req.query.Rejected

  PQ5 = PQ5.replace(/[^a-zA-Z ]/g, "")
  res.send("success")
});

io.on('connection', (socket) =>  {
  socket.on('chat message', msg => {
    io.emit('chat message', msg);
  });
});  

function mysql_real_escape_string (str) {
  if(typeof str == 'undefined'){
    return "undefined"
  }else{
    return str.replace(/[\0\x08\x09\x1a\n\r"'\\\%]/g, function (char) {
      switch (char) {
          case "\0":
              return "\\0";
          case "\x08":
              return "\\b";
          case "\x09":
              return "\\t";
          case "\x1a":
              return "\\z";
          case "\n":
              return "\\n";
          case "\r":
              return "\\r";
          case "\"":
          case "'":
          case "\\":
          case "%":
              return "\\"+char; // prepends a backslash to backslash, percent,
                                // and double/single quotes
          default:
              return char;
      }
    });
  }
}

app.get('/feedback', function(req, res){
  const MTurkId = req.query.MTurkId
  const UserId = req.query.UserId
  const EmailId = req.query.EmailId
  const Trial = String(req.query.Trial)
  const Condition = parseInt(req.query.Condition)
  const Messages = req.query.Messages
  const relevantQuestions = req.query.RelevantQuestions
  const totalQuestions = req.query.TotalQuestions
  const InitialMessage = req.query.InitialMessage
  const stringify = JSON.stringify(Messages[Messages.length - 1])
  if (typeof stringify == 'undefined'){
    res.send("I couldn't provide feedback for that. My appologies.")
  }
  var sql_message = mysql_real_escape_string(stringify)
  
  var response = ""
  if(relevantQuestions < 2 && totalQuestions < 10){
    var last_message = Messages[Messages.length - 1]
    var last_message_relevant = [{"role": "system", "content": "Your job is to determine if messages sent by students in a phishing email identification experiment are about the topic of emails or phishing emails. Reply with a 1 if the message is relevant to phishing emails or a 0 if it is irrelevant."}, last_message]
    //openAPIclient.chat.completions(last_message_relevant, {max_tokens:1})
    let messages = last_message_relevant
    //openAPIclient.chat.completions.create({last_message_relevant, model:'gpt-4o-mini', max_tokens: 1})
    openAPIclient.chat.completions.create({messages, model:'gpt-4o-mini', max_tokens: 150})
    .then(result => {
        if(result.choices[0].message.content == "0" & InitialMessage != "true"){
          response = "That message does not seem to be a question relevant to phishing emails, please ask a relevant question or continue to the next trial of the experiment." 
          res.send(response)
        }else if(last_message.content.length < 20 & InitialMessage != "true"){
          response = "That message seems too short for me to reply with useful information about phishing emails. Please ask a question with more context so I can be a helpful teaching aid."
          res.send(response)
        }else{
          let messages = Messages
          openAPIclient.chat.completions.create({messages, model:'gpt-4o-mini', max_tokens: 150})
          .then(result => {
              response = mysql_real_escape_string(result.choices[0].message.content)
              res.send(result.choices[0].message.content)
            }
          ).catch(err => {
            console.log(err)
            response = "I couldn't provide feedback for that. My appologies."
            res.send(response)
          });
        }
      }
    ).catch(err => {
      console.log(err)
      response = "I couldn't provide feedback for that. My appologies."
      res.send(response)
    });
  }else{
    res.send("")
  }

});
  
app.get('/queryIBL', function(req, res){
  res.send("")
});

app.get('/', (req,res)=>{
  res.sendFile(path.join(__dirname, 'public/html/landing.html'));
})

app.get('/experiment1', (req,res)=>{
  res.sendFile(path.join(__dirname, 'public/html/startExperiment1.html'));
})

app.get('/experiment2', (req,res)=>{
  res.sendFile(path.join(__dirname, 'public/html/startExperiment2.html'));
})

app.get('/experiment3', (req,res)=>{
  res.sendFile(path.join(__dirname, 'public/html/startExperiment3.html'));
})

app.get('/experiment4', (req,res)=>{
  res.sendFile(path.join(__dirname, 'public/html/startExperiment4.html'));
})

const port = process.env.PORT || 8981;
http.listen(port, () => console.log(`Listening on port ${port}`));
