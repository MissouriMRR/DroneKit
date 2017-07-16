new Vue({

  el: '#app',
  
  data: {
    app: null,
    loading_message: 'Connecting to server...',
    socket: null,
    battery_percentage: 85,
    altitude: null,
    attitude: null
  },

  created: function()
  {
    socket = io.connect('http://' + document.domain + ':' + location.port);
  },

  mounted: function()
  {
    app = this;

    socket.on('connect', function() {
        console.log("Connected to server...");
    });

    socket.on('status', function(data) {
      app.loading_message = "Updating vehicle status...";

      console.log("Updating vehicle status...");

      // loading_screen.style.display = "none";
      // vehicle_info.style.display = "block";

    });

  }

})