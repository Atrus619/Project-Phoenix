var app = require('express')();
var http = require('http').createServer(app);
var user_io = require('socket.io')(http, {
  // 'pingInterval': Infinity,
  // 'pingTimeout': Infinity
});

// var eventDebug = require('event-debug')
// eventDebug(http, 'MyServer')

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

user_io.on('connection', function(socket){
  console.log('a user connected');
  socket.on('disconnect', function(){
    console.log('user disconnected');
  });
  socket.on('connect', function(){
    console.log('CONNECT!')
    user_io.emit('user connect', '')
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

// Run normally: nodemon app/index.js
// Debug: DEBUG=engine* nodemon app/index.js
// Restart during: type `rs`
