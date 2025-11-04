//import {initJsPsych} from 'jspsych'
var jsPsych = initJsPsych();

function getJsPsych(){
  return jsPsych
}

//import JsPsych from "./app.mjs";
//var JsPsych = JsPsych

var TODAY = new Date();
var DD = String(TODAY.getDate()).padStart(2, '0');
var MM = String(TODAY.getMonth() + 1).padStart(2, '0');
var YYYY = TODAY.getFullYear();
const DATE = YYYY + MM + DD;

var conditionArray = [1]
var Condition = 1
/*
  Main Study
  Condition 0: Randomly selected emails with point feedback: already completed
  Condition 1: IBL+LLM selected emails with point feedback 
  Condition 2: Randomly selected emails with GPT natural language feedback 
  Condition 3: IBL+LLM selected emails with IBL+LLM informed natural language feedback
*/


var timeline = [];

var userID = Math.floor(Math.random() * 1e9);

var check_mid = function(elem) {
  var mid = document.querySelector('input[name="mid"]').value
  if(mid.length < 6){
    alert("Your ID doesn't appear to be long enough.");
    return false
  }else{
    jsPsych.data.addProperties( {mid: document.querySelector('input[name="mid"]').value})
    return true 
  }
};

var check_testing = function(elem) {
  jsPsych.data.addProperties( { 
    mid: 1, 
    uid: userID,
    consent: 1,
    age: 22,
    gender:1, 
    education: 1,
    victim: 1,
    chatbot:1,
    country: 1});
  jsPsych.data.addProperties({q0: 1})
  jsPsych.data.addProperties({q1: 1})
  jsPsych.data.addProperties({q2: 1})
  jsPsych.data.addProperties({q3: 1})
  jsPsych.data.addProperties({q4: 1})
  jsPsych.data.addProperties({q5: 1})

  switch (document.querySelector(".condition select").value){
    case 1:
      jsPsych.data.addProperties( {condition: 1})
    case 2:
      jsPsych.data.addProperties( {condition: 2})
    case 3:
      jsPsych.data.addProperties( {condition: 3})
    case 4:
      jsPsych.data.addProperties( {condition: 4})
  }
  // jsPsych.data.write(data_object)
  return true;
};

//consent form 
// sample function that might be used to check if a subject has given
// consent to participate.
var check_consent = function(elem) {
  if (document.getElementById('EligibilityYes').checked && document.getElementById('ParticipateYes').checked && document.getElementById('UnderstandYes').checked && document.getElementById('EighteenYes').checked) {
      return true;
  }
  else if(document.getElementById('EligibilityNo').checked || document.getElementById('ParticipateNo').checked || document.getElementById('UnderstandNo').checked || document.getElementById('EighteenNo').checked){
    var mid = jsPsych.data.get().first(1)?.values()[0]?.mid.toString()
    window.location.replace("http://foundationphishing.eastus.cloudapp.azure.com:8888/reject/" + mid);
    return false;
}else{
  alert("Please respond to each question.")
  return false
}
  return false;
};

var consent = {
  type: jsPsychExternalHtml,
  url: "html/consent.html",
  cont_btn: "Continue",
  check_fn: check_consent
};

var mid = {
  type: jsPsychExternalHtml,
  url: "html/mid.html",
  cont_btn: "Submit",
  check_fn: check_mid
};

var testing = {
  type: jsPsychExternalHtml,
  url: "html/testingConsent.html",
  cont_btn: "Continue",
  check_fn: check_testing
}


var debrief = {
  type: jsPsychExternalHtml,
  url: "html/debrief.html",
  cont_btn: "Continue",
};

var enter_fullscreen = {
  type: jsPsychFullscreen,
  fullscreen_mode: true
};

var check_quiz = function(data) {
  checked_0_0 = document.getElementById('jspsych-survey-multi-choice-response-0-0').checked
  checked_0_1 = document.getElementById('jspsych-survey-multi-choice-response-0-1').checked
  checked_0_2 = document.getElementById('jspsych-survey-multi-choice-response-0-2').checked
  checked_0_3 = document.getElementById('jspsych-survey-multi-choice-response-0-3').checked
  var q0 = [checked_0_0, checked_0_1, checked_0_2, checked_0_3]

  checked_1_0 = document.getElementById('jspsych-survey-multi-choice-response-1-0').checked
  checked_1_1 = document.getElementById('jspsych-survey-multi-choice-response-1-1').checked
  checked_1_2 = document.getElementById('jspsych-survey-multi-choice-response-1-2').checked
  checked_1_3 = document.getElementById('jspsych-survey-multi-choice-response-1-3').checked
  var q1 = [checked_1_0, checked_1_1, checked_1_2, checked_1_3]

  checked_2_0 = document.getElementById('jspsych-survey-multi-choice-response-2-0').checked
  checked_2_1 = document.getElementById('jspsych-survey-multi-choice-response-2-1').checked
  checked_2_2 = document.getElementById('jspsych-survey-multi-choice-response-2-2').checked
  checked_2_3 = document.getElementById('jspsych-survey-multi-choice-response-2-3').checked
  var q2 = [checked_2_0, checked_2_1, checked_2_2, checked_2_3]

  checked_3_0 = document.getElementById('jspsych-survey-multi-choice-response-3-0').checked
  checked_3_1 = document.getElementById('jspsych-survey-multi-choice-response-3-1').checked
  checked_3_2 = document.getElementById('jspsych-survey-multi-choice-response-3-2').checked
  checked_3_3 = document.getElementById('jspsych-survey-multi-choice-response-3-3').checked
  var q3 = [checked_3_0, checked_3_1, checked_3_2, checked_3_3]

  checked_4_0 = document.getElementById('jspsych-survey-multi-choice-response-4-0').checked
  checked_4_1 = document.getElementById('jspsych-survey-multi-choice-response-4-1').checked
  checked_4_2 = document.getElementById('jspsych-survey-multi-choice-response-4-2').checked
  checked_4_3 = document.getElementById('jspsych-survey-multi-choice-response-4-3').checked
  var q4 = [checked_4_0, checked_4_1, checked_4_2, checked_4_3]

  checked_5_0 = document.getElementById('jspsych-survey-multi-choice-response-5-0').checked
  checked_5_1 = document.getElementById('jspsych-survey-multi-choice-response-5-1').checked
  checked_5_2 = document.getElementById('jspsych-survey-multi-choice-response-5-2').checked
  checked_5_3 = document.getElementById('jspsych-survey-multi-choice-response-5-3').checked
  var q5 = [checked_5_0, checked_5_1, checked_5_2, checked_5_3]
  
  if(q0.includes(true) && q1.includes(true) && q2.includes(true) && q3.includes(true) && q4.includes(true) && q5.includes(true)){
    jsPsych.data.addProperties({q0: q0.indexOf(true)})
    jsPsych.data.addProperties({q1: q1.indexOf(true)})
    jsPsych.data.addProperties({q2: q2.indexOf(true)})
    jsPsych.data.addProperties({q3: q3.indexOf(true)})
    jsPsych.data.addProperties({q4: q4.indexOf(true)})
    jsPsych.data.addProperties({q5: q5.indexOf(true)})
    return true
  }
  else {
      alert("Please respond to each question.");
      return false;
  }
}

var check_demographics = function(elem) {
  if (document.getElementById('ParticipateYes').checked && document.getElementById('PrivateYes').checked && document.getElementById('CompleteYes').checked && document.querySelector('input[name="age"]').value != "" && document.querySelector(".gender select").value != "Please select." && document.querySelector(".education select").value != "Please select the highest level of education that you have achieved/are currently working towards." && document.querySelector(".victim select").value != "Please select." && document.querySelector(".country select").value != "Please select the country you are currently in.") {
      jsPsych.data.addProperties( { 
        uid: userID,
        consent: document.getElementById('ParticipateYes').checked,
        age: document.querySelector('input[name="age"]').value,
        gender:document.querySelector(".gender select").value, 
        education: document.querySelector(".education select").value,
        victim: document.querySelector(".victim select").value,
        chatBot: document.querySelector(".chatBot select").value,
        country: document.querySelector(".country select").value })
      
      var MTurkId     = jsPsych.data.get().last(1)?.values()[0]?.mid.toString()
      var Userid      = jsPsych.data.get().last(1)?.values()[0]?.uid.toString()
      var Consent     = jsPsych.data.get().last(1)?.values()[0]?.consent.toString()
      var Age         = jsPsych.data.get().last(1)?.values()[0]?.age.toString()
      var Gender      = jsPsych.data.get().last(1)?.values()[0]?.gender.toString()
      var Education   = jsPsych.data.get().last(1)?.values()[0]?.education.toString()
      var Victim      = jsPsych.data.get().last(1)?.values()[0]?.victim.toString()
      var Chatbot      = jsPsych.data.get().last(1)?.values()[0]?.chatBot.toString()
      var Country     = jsPsych.data.get().last(1)?.values()[0]?.country.toString()

      $.ajax({
          url: "/questionnaire",
          data: { 
          "MTurkId"   : MTurkId,
          "Age"       : Age,
          "Gender"    : Gender,
          "Education" : Education,
          "Country"   : Country,
          "Victim"    : Victim,
          "Chatbot"   : Chatbot,
          "Consent"   : Consent, 
          "Q0"        : 0,
          "Q1"        : 0,
          "Q2"        : 0,
          "Q3"        : 0,
          "Q4"        : 0,
          "Q5"        : 0,
          "Userid"    : Userid,
          "PQ0"       : "PQ0",
          "PQ1"       : "PQ1",
          "PQ2"       : "PQ2",
          "PQ3"       : "PQ3",
          "PQ4"       : "PQ4",
          "PQ5"       : "PQ5",
          "Rejected"  : 0,
          },
          async:false,
          cache: false,
          type: "GET",
          success: function(response) {
          },
          error: function(xhr) {
          console.log(xhr)
        }
      })
        
      return true;
  }
  else if(document.getElementById('ParticipateNo').checked || document.getElementById('PrivateNo').checked || document.getElementById('CompleteNo').checked){
      var mid = jsPsych.data.get().first(1)?.values()[0]?.mid.toString()
      window.location.replace("http://foundationphishing.eastus.cloudapp.azure.com:8888/reject/" + mid);
      return false;
  }else{
    alert("Please respond to each question.")
    return false
  }
  return false;
};

var demographics = {
  type: jsPsychExternalHtml,
  url: "html/demographics.html",
  cont_btn: "Continue",
  check_fn: check_demographics,
};

var instructions = {
  type: jsPsychExternalHtml,
  url: "html/instructions.html",
  cont_btn: "Continue"
};

var quiz = {
  type: jsPsychExternalHtml,
  url: "html/quiz.html",
  cont_btn: "Continue",
  check_fn: check_quiz,
};

var paymentInstructions = {
  type: jsPsychExternalHtml,
  url: "html/paymentInstructions.html",
  cont_btn: "Continue"
};

var preTrialInstructions = {
  type: jsPsychExternalHtml,
  url: "html/preTrialInstructions.html",
  cont_btn: "Continue"
};

var chatbotInstructions = {
  type: jsPsychExternalHtml,
  url: "html/chatbotInstructions.html",
  cont_btn: "Continue"
}

var postTrialInstructions = {
  type: jsPsychExternalHtml,
  url: "html/postTrialInstructions.html",
  cont_btn: "Continue"
};

var check_postQuestionnaire = function(elem) {
  var base1 = "Please select the percentage that best fits your experience in the study."
  var base2 = "What criteria did you use to identify whether an email was a phishing attempt?"
  if(document.getElementById('phishingWriting').value != base1 && 
  document.getElementById('hamWriting').value != base1 &&
  document.getElementById('phishingStyling').value != base1 &&
  document.getElementById('hamStyling').value != base1 &&
  document.getElementById('openResponse').value != base2){
    jsPsych.data.addProperties( { 
      uid: userID,
      phishingWriting: document.getElementById('phishingWriting').value,
      hamWriting: document.getElementById('hamWriting').value,
      phishingStyling:document.getElementById('phishingStyling').value, 
      hamStyling: document.getElementById('hamStyling').value,
      openResponse: document.getElementById('openResponse').value})
      
      var MTurkId     = jsPsych.data.get().last(1)?.values()[0]?.mid.toString()
      var Userid      = jsPsych.data.get().last(1)?.values()[0]?.uid.toString()
      var Consent     = jsPsych.data.get().last(1)?.values()[0]?.consent.toString()
      var Age         = jsPsych.data.get().last(1)?.values()[0]?.age.toString()
      var Gender      = jsPsych.data.get().last(1)?.values()[0]?.gender.toString()
      var Education   = jsPsych.data.get().last(1)?.values()[0]?.education.toString()
      var Victim      = jsPsych.data.get().last(1)?.values()[0]?.victim.toString()
      var Chatbot      = jsPsych.data.get().last(1)?.values()[0]?.chatBot.toString()
      var Country     = jsPsych.data.get().last(1)?.values()[0]?.country.toString()
      var Q0          = jsPsych.data.get().last(1)?.values()[0]?.q0
      var Q1          = jsPsych.data.get().last(1)?.values()[0]?.q1
      var Q2          = jsPsych.data.get().last(1)?.values()[0]?.q2
      var Q3          = jsPsych.data.get().last(1)?.values()[0]?.q3
      var Q4          = jsPsych.data.get().last(1)?.values()[0]?.q4
      var Q5          = jsPsych.data.get().last(1)?.values()[0]?.q5
      var PQ0         = "Unused"
      var PQ1         = jsPsych.data.get().last(1)?.values()[0]?.phishingWriting.toString()
      var PQ2         = jsPsych.data.get().last(1)?.values()[0]?.hamWriting.toString()
      var PQ3         = jsPsych.data.get().last(1)?.values()[0]?.phishingStyling.toString()
      var PQ4         = jsPsych.data.get().last(1)?.values()[0]?.hamStyling.toString()
      var PQ5         = jsPsych.data.get().last(1)?.values()[0]?.openResponse.toString()
      var Rejected    = 0

      $.ajax({
          url: "/questionnaire",
          data: { 
          "MTurkId"   : MTurkId,
          "Age"       : Age,
          "Gender"    : Gender,
          "Education" : Education,
          "Country"   : Country,
          "Victim"    : Victim,
          "Chatbot"   : Chatbot,
          "Consent"   : Consent, 
          "Q0"        : Q0,
          "Q1"        : Q1,
          "Q2"        : Q2,
          "Q3"        : Q3,
          "Q4"        : Q4,
          "Q5"        : Q5,
          "Userid"    : Userid,
          "PQ0"       : PQ0,
          "PQ1"       : PQ1,
          "PQ2"       : PQ2,
          "PQ3"       : PQ3,
          "PQ4"       : PQ4,
          "PQ5"       : PQ5,
          "Rejected"  : Rejected,
          },
          async:false,
          cache: false,
          type: "GET",
          success: function(response) {
          },
          error: function(xhr) {
          console.log(xhr)
        }
      })
    
    
    return true
  }else{
    alert("Please respond to each question.")
    return false
  }
};

// Define a trial for the post-questionnaire stage
var postQuestionnaire = {
  type: jsPsychExternalHtml,
  url: "html/postQuestionnaire.html",
  cont_btn: "Continue",
  check_fn: check_postQuestionnaire,
};


// Define a trial for the end stage
var endTrial = {
    type: jsPsychHtmlKeyboardResponse,
    choices: "NO_KEYS",
    stimulus: function(data){
      
      var condition = jsPsych.data.get().last(2)?.values()[0]?.condition
      var conditon_string = "In this experiment, you observed emails that were written by a human expert, and styled using a generative AI model."
      var num_correct_trials = jsPsych.data.get().filter({correct: 1}, {trial_type: "external-html"}).count() / 2;
      var num_incorrect_trials = jsPsych.data.get().filter({correct: 0}, {trial_type: "external-html"}).count() / 2;
      var payout = (0.2 * num_correct_trials) - (0.2 * num_incorrect_trials)
      if(payout < 0){
        payout = 0
      }
      payout = payout.toFixed(2)
      payout = payout.toString()
      // Save all data to csv as a backup to the database: 
      // console.log(all_data.csv());
      var output = `<h2> End of demo. Thank you for your interest </h2>`
      //console.log(jsPsych.data.get().csv()); 
      // console.log(jsPsych.data.get())
      return output
    },
    trial_duration: 1000000,  // Adjust the duration as needed
    cont_btn: "Continue",
};


var check_experiment_data = function(elem) {
  button_checked = $('#btnradioNo')[0].checked || $('#btnradioYes')[0].checked
  confidences = [$('#rating1')[0].checked, $('#rating2')[0].checked, $('#rating3')[0].checked, $('#rating4')[0].checked, $('#rating5')[0].checked]
  actions = [$('#action1')[0].checked, $('#action2')[0].checked, $('#action3')[0].checked, $('#action4')[0].checked, $('#action5')[0].checked, $('#action6')[0].checked]
  if(button_checked && confidences.some(function(e){return e}) && actions.some(function(e){return e})){
    var Correct = 0
    var Type = jsPsych.data.get().last(1)?.values()[0]?.type
    emailDecision = false 
    if($('#btnradioYes')[0].checked){
      emailDecision = true 
    }
    howConfident = confidences.indexOf(true)
    action = actions.indexOf(true)
    if((Type == "phishing" && emailDecision == true) || (Type == "ham" && emailDecision == false)){
        Correct = 1
    }
    //console.log("Correct " + Correct, " type " + Type + " emailDecision " + emailDecision)
    // Remove either this one or the other one, dont need 2 it is messing up the bonus calculation
    jsPsych.data.addProperties( {
        emailDecision: emailDecision,
        howConfident: howConfident + 1,
        action: action + 1,
        type: Type,
        correct: Correct
    });

    return true 
  } else {
    alert("If you wish to continue, you must select a decision, a confidence level, and an action.");
    return false 
  }   
};

var check_feedback_data = function(elem){
  if(localStorage.getItem("Done") == 0){
    if(Condition != 1){
      alert("You responded incorrectly to whether this was a phishing email or not. If you wish to continue, you must ask the teacher at least one relevant question.")
    }else{
      alert("You responded incorrectly to whether this was a phishing email or not. If you wish to continue, you must ask the teacher at least one relevant question.")
    }
    return false
  }else{
    return true
  }
}

// Display the previous email shown to the participant
function displayEmail(){
  return jsPsych.data.get().last(1)?.values()[0]?.email;
}

function shuffle(array) {
  let currentIndex = array.length;

  // While there remain elements to shuffle...
  while (currentIndex != 0) {

    // Pick a remaining element...
    let randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }
  return array
}

function shuffleArray(array) {
  for (var i = array.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var temp = array[i];
      array[i] = array[j];
      array[j] = temp;
  }
  return array
}

function experiment(phaseValue, repeat) {
  var email_types = []
  var pretrain_types = []
  var train_types = []
  var feedbackLength = 20
  var noFeedbackLength = 5
  if(phaseValue == "preTraining" || phaseValue == "postTraining" ){
    for (let i = 0; i < noFeedbackLength; i++) {
      pretrain_types.push("ham")
      pretrain_types.push("spam")
    }
    pretrain_types = shuffleArray(pretrain_types)
    pretrain_types = shuffle(pretrain_types)
    pretrain_types = shuffleArray(pretrain_types)
    pretrain_types = shuffle(pretrain_types)
    email_types = email_types.concat(pretrain_types);
  }else{
    for (let i = 0; i < feedbackLength; i++) {
      train_types.push("ham")
      train_types.push("spam")
    }
    train_types = shuffle(train_types)
    train_types = shuffleArray(train_types)
    train_types = shuffle(train_types)
    train_types = shuffleArray(train_types)
    email_types = email_types.concat(train_types);
  }
  // Initialize a counter variable
  var trialNumber = 0;
  var trials = []
  stimuli = {
    type: jsPsychExternalHtml,
    url: "/html/stimuli.html",
    cont_btn: "nextButton",
    check_fn: check_experiment_data,
    data: {
      userId: userID,
      condition: Condition,
      phase: phaseValue,
      trial: null,
      type: null,
      previouslyObserved: "-3,-2,-1",
    },
    execute_script: true,
    on_start: function (trial) {
      // Access trial data here
      trialNumber++;
      //console.log("trialNumber: ", trialNumber, " type: ", email_types[trialNumber], " all trial types ", email_types, "phaseValue: ", phaseValue)
      trial.data.trial = trialNumber;
      trial.data.type = email_types[trialNumber-1]
      jsPsych.data.get().push(trial.data);
    },
  }
  trials.push(stimuli)
  if(phaseValue == "Training"){
    feedback = {
      type: jsPsychExternalHtml,
      url: "/html/feedback.html",
      cont_btn: "nextButton",
      check_fn: check_feedback_data,
      data: {
        userId: userID,
        email: displayEmail,
        condition: Condition,
        phase: phaseValue,
        trial: null,
        type: null,
        previouslyObserved: "-3,-2,-1",
      },
      execute_script: true,
      on_start: function (trial) {
        // Access trial data here
        trial.data.trial = trialNumber;
        trial.data.type = email_types[trialNumber-1]
        jsPsych.data.get().push(trial.data);
      },
    }
    trials.push(feedback)
  }
  return {
    timeline: trials,
    repetitions: repeat,
    randomize_order: false
  };
  }


TESTING = 1
// Create a timeline with the defined trials

if(TESTING != 1){
  timeline.push(mid)
  timeline.push(consent)
  timeline.push(demographics)
  timeline.push(instructions)
  timeline.push(quiz);
  timeline.push(paymentInstructions)
  timeline.push(experiment('preTraining', 10));
  if(Condition != 1){
    timeline.push(chatbotInstructions)
  }else{
    timeline.push(preTrialInstructions)
  }
}

timeline.push(experiment('Training', 10));
timeline.push(endTrial);


function startExp() {
  // Initialize the PsychJS experiment with the timeline
  jsPsych.run(timeline);
  //return true;
}