new Vue({

  el: '#app',
  
  data: {
    HEARBEAT_TIMEOUT_SECONDS: 59,
    loading_message: 'Connecting to server...',
    socket: null,
    altitude: {},
    attitude: {},
    vehicle: {},
    vehicle_data: null,
    hearbeat_timer_id: null,
    time_since_last_update: null,
    alert_visible: false,
    available_vehicles: []
  },

  methods: {

    setVehicle: function()
    {
      app = this; 

      socket.emit("initialization", app.vehicle);
      app.loading_message = "Connecting to " + app.vehicle + "...";
      vehicle_selector.style.display = "none";
      console.log("Sent vehicle selection to server.");
      NProgress.start()
    },

    calculateTimeDelta: function()
    {
      app = this;

      now = new Date();
      seconds_elapsed = Math.round(now - app.time_since_last_update / 1000) % 60;

      if(app.time_since_last_update != null && seconds_elapsed >= app.HEARBEAT_TIMEOUT_SECONDS)
      {
        if(!app.alert_visible)
          {
            notie.alert({type: 'warning', text: 'Attempting to reconnect to vehicle...' , stay: true})
            app.alert_visible = true;
          }
      }
      else if(seconds_elapsed <= app.HEARBEAT_TIMEOUT_SECONDS)
      {
          if(app.alert_visible)
          {
            notie.hideAlerts();
            app.alert_visible = false;
          }
      }
    },

    setupHeartbeatThread: function()
    {
      console.log("Updated vehicle status.");

      setTimeout(function() {
        loading_screen.style.display = "none";
        information_view.style.display = "block";
        NProgress.done()

        app.hearbeat_timer_id = setInterval(function() {
          socket.emit("update_status");

        }, 1000);

      }, 1500);
    }

  },

  created: function()
  {
    NProgress.start();
    socket = io.connect('http://' + document.domain + ':' + location.port);

    window.onbeforeunload = function()
    {
      socket.emit("shutdown");
      console.log("Sent vehicle shutdown to server.");
    };

  },

  mounted: function()
  {
    app = this;

    socket.on('connect', function() {
        console.log("Connected to server.");
        app.loading_message = "Fetching list of vehicles...";
    });

    socket.on('vehicle_update', function(data) {
      
      NProgress.done()
      if(data != null)
      {
        console.log("Retrieved vehicle list.")
        app.available_vehicles = data;
        app.loading_message = "Select a vehicle";
      }
      else 
      {
        vehicle_selector.style.display = "none";
        app.loading_message = "Something went wrong server side, please restart the app and try again.";
      }

    });

    socket.on('information', function(data) {
      console.log(data);
    });

    socket.on('status', function(data) {
      app.loading_message = "Updating vehicle status...";
      app.vehicle_data = data;
      app.vehicle_data.uptime_moment = moment.utc(data.uptime);
      app.time_since_last_update = new Date(app.vehicle_data.hearbeat);

      // console.log(app.vehicle_data.uptime_moment)

      app.calculateTimeDelta();
      

        if(information_view.style.display == "none")
        {
          //First time we've gotten the vehicle's status.
          app.setupHeartbeatThread();
        }

    });

  }

})