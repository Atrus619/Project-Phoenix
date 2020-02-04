var app = require('express')();
var http = require('http').createServer(app);
var user_io = require('socket.io')(http);

var chatbot_server = require('net');
var chatbot_client = new chatbot_server.Socket();
chatbot_client.connect(8765, '127.0.0.1', function(){
  console.log('listening for chatbot inputs on *:8765')
});
chatbot_client.on('msg', function(msg){
  console.log('Bot:', msg)
})

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
});

http.listen(3000, function(){
  console.log('listening for user inputs on *:3000');
});

// Trying these links:
// https://stackoverflow.com/questions/39184455/connect-js-client-with-python-server
// https://ianhinsdale.com/post/communicating-between-nodejs-and-python/
