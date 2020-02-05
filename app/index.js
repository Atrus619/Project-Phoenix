var app = require('express')();
var http = require('http').createServer(app);
var user_io = require('socket.io')(http);

var eventDebug = require('event-debug')
eventDebug(http, 'MyServer')

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

user_io.on('connection', function(socket){
  console.log('a user connected');
  socket.on('disconnect', function(){
    console.log('user disconnected');
  });
  socket.on('chat message', function(msg){
    console.log('User:', msg)
    user_io.emit('chat message', msg);
  });
  socket.on('message', function(event){
    console.log('got something!')
  });
  socket.on('error', function(error){
    console.log('Error:', error)
  })
  eventDebug(socket, 'socket')
});

http.listen(3000, function(){
  console.log('listening for user inputs on *:3000');
});

// Trying these links:
// https://stackoverflow.com/questions/39184455/connect-js-client-with-python-server
// https://ianhinsdale.com/post/communicating-between-nodejs-and-python/
