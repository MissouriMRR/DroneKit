import sys
import yaml
import time
from datetime import datetime

from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from json import loads, dumps

sys.path.append('../../Flight/')

from AirTrafficControl import Tower

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

tower = Tower()

CONFIG_FILENAME = "drone_configs.yml"

vehicle_config_data = None

def load_config_file():
    try:
        global vehicle_config_data 
        config_file = open(CONFIG_FILENAME, 'r')
        vehicle_config_data = yaml.load(config_file)
        config_file.close()
    except IOError:
        print("\nFailed to get configuration file, some information may not be available.\n")

def get_battery():
    battery_info = {}
    battery_info["voltage"] = tower.vehicle.battery.voltage
    battery_info["full_voltage"] = vehicle_config_data["battery"]["full_voltage"]
    battery_info["percent_remaining"] = tower.vehicle.battery.voltage / vehicle_config_data["battery"]["full_voltage"]

    return battery_info

def fake_get_battery():
    battery_info = {}
    battery_info["voltage"] = 11.34
    battery_info["full_voltage"] = vehicle_config_data["battery"]["full_voltage"]
    battery_info["percent_remaining"] = 11.34 / vehicle_config_data["battery"]["full_voltage"]

    return battery_info

def get_vehicle_status():
    """Laundry list function.

    """
    if not vehicle_config_data:
        return
    status = {}
    status['battery'] = fake_get_battery()
    status['hearbeat'] = int(time.mktime(datetime.utcnow().timetuple())) * 1000
    return status

@app.route('/')
def index():
    return render_template("index.html")

@socketio.on('connect')
def on_connect():
    emit('information', 'Initialization in progress...')
    load_config_file()
    emit('vehicle_update', vehicle_config_data, json=True)

@socketio.on('update_status')
def on_status():
    emit('status', get_vehicle_status())

@socketio.on('initialization')
def on_initialization(selected_vehicle_name):
    global vehicle_config_data
    for vehicle in vehicle_config_data:
        if vehicle["name"] == selected_vehicle_name:
            vehicle_config_data = vehicle
            emit('status', get_vehicle_status())
            return

if __name__ == '__main__':
    socketio.run(app, debug=True)
