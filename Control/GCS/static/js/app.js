new Vue({

  el: '#app',
  
  data: {
    loading_message: 'Connecting to server...',
    socket: null,
    battery_percentage: 85
  },

  created: function()
  {
    socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('connect', function(data) {
        console.log("Connected to server...");
        loading_message = data;
    });

    socket.on('status', function(data) {
      console.log(data);
    });
  },

  mounted: function()
  {



  }

})