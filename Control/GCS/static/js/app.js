new Vue({

  el: '#app',
  
  data: {
    message: 'Hello Vue!',
    socket: null
  },

  created: function()
  {
    socket = io.connect('http://' + document.domain + ':' + location.port);
    socket.on('connect', function() {
        socket.emit('my event', {data: 'I\'m connected!'});
    });
  },

  mounted: function()
  {


    console.log("Microsoft office.");
  }

})