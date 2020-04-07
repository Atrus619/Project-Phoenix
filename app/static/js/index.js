var recognition = new webkitSpeechRecognition();

recognition.onresult = function(event) {
  var saidText = "";
  for (var i = event.resultIndex; i < event.results.length; i++) {
    if (event.results[i].isFinal) {
      saidText = event.results[i][0].transcript;
    } else {
      saidText += event.results[i][0].transcript;
    }
  }

  if (saidText == "send"){
    sendMessage();
  } else{
    document.getElementById('m').value += saidText;
  }
}

function startRecording(){
  recognition.start();
}

var socket = io();

sendMessage = function(){
  if ($('#m').val() != ''){
    socket.emit('user message', $('#m').val());
    $('#m').val('');
  }
  return false;
}

$(function(){
  $("#chatForm").bind('keypress', function(event){
    if (event.which == 13){
      event.preventDefault();
      sendMessage();
    };
  })
})

socket.on('user message', function(msg){
  var newUserMsg = $('<li class="userMsg">').text(msg);
  $('#messages').append(newUserMsg);
});

// This is implemented differently to allow links to be passed from the bot
socket.on('bot message', function(msg){
  var newBotMsg = '<li class="botMsg">'.concat(msg).concat("</li>");
  $('#messages').append(newBotMsg);
});

loadContent = function(link){
  $("#viz").attr("src", link);
  return false;
}

resizeRightColumn = function(){
  var headerHeight = $('.header').outerHeight();
  var currentHeight = $( window ).height();


  var currentWidth = $( document ).width();
  if (currentWidth <= 800){
    var formHeight = $('#chatForm').height();
    $('.rightColumn').height(currentHeight - headerHeight - formHeight);
    $('.rightColumn').css('padding-bottom', formHeight);
  } else{
    $('.rightColumn').height(currentHeight - headerHeight);
    $('.rightColumn').css('padding-bottom', 0);
  };
};

resizeLogo = function(){
  var headerHeight = $('.header').height();
  $('.logo').height(headerHeight);
};

$( document ).ready(function() {
  resizeRightColumn();
  resizeLogo();
});

$( window ).resize(function() {
  resizeRightColumn();
  resizeLogo();
});
