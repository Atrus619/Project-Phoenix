var express = require('express');
var app = express();

app.get('/',function(req, res){
  res.sendFile(__dirname + '/index.html');
});

var server = require('http').Server(app);
io = require('socket.io')(server);
users = {};
server.listen(3000, function(){
  console.log('listening on *:3000')
});

io.sockets.on('connection', function(socket){
  console.log('user connected');

  socket.on('new', function(data, callback){
    console.log(data.name);

    if(data in users)
      callback(false);
    else{
      callback(true);
      socket.name = data.name
      users[socket.name] = socket;
    }
  })

  socket.on('msg', function(data, callback){
    callback(data.msg);
    io.to(users[data.to].emit('priv', data.msg))
  })

  socket.on('disconnect', function(){
    console.log('user disconnected')
  })
})
