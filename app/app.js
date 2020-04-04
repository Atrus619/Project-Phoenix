var express = require('express');
var app = express();
var http = require('http').createServer(app);
var user_io = require('socket.io')(http);
var path = require('path');

// var eventDebug = require('event-debug')
// eventDebug(http, 'MyServer')

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

app.use(express.static(path.join(__dirname, 'static')))

user_io.on('connection', function(socket){
  console.log('a user connected');
  if (Object.keys(user_io.sockets.clients().connected).length > 1){
    console.log('2 users are connected!');
    user_io.emit('user connect', '')
  }
  socket.on('disconnect', function(){
    console.log('user disconnected');
  });
  socket.on('user message', function(msg){
    console.log('User:', msg)
    user_io.emit('user message', msg);
  });
  socket.on('bot message', function(msg){
    console.log('Chatbot:', msg)
    user_io.emit('bot message', msg);
  });
  // eventDebug(socket, 'socket')
});

http.listen(3000, function(){
  console.log('listening for user inputs on *:3000');
});

// Run normally: nodemon app/app.js
// Debug: DEBUG=engine* nodemon app/app.js
// Restart during: type `rs`
