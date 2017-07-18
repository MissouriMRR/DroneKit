import sys
import yaml
import time

from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from json import loads, dumps
from math import degrees
from datetime import datetime

sys.path.append('../../Flight/')

from AirTrafficControl import Tower

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

tower = None

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
    global vehicle_config_data 
    battery_info = {}
    battery_info["voltage"] = tower.vehicle.battery.voltage
    battery_info["full_voltage"] = vehicle_config_data["battery"]["full_voltage"]
    battery_info["failsafe_voltage"] = vehicle_config_data["battery"]["failsafe_voltage"]
    current = tower.vehicle.battery.voltage - battery_info["failsafe_voltage"]
    full = battery_info["full_voltage"] - battery_info["failsafe_voltage"]
    battery_info["percent_remaining"] = (current / full if battery_info["voltage"] > battery_info["failsafe_voltage"] else 0.00)

    return battery_info

def get_velocities():
    velocities = {}
    velocities['x'] = tower.vehicle.velocity[0]
    velocities['y'] = tower.vehicle.velocity[1]
    velocities['z'] = tower.vehicle.velocity[2]

    return velocities

def get_attitude():
    attitude_deg = {}
    attitude_deg["roll"] = degrees(tower.vehicle.attitude.roll)
    attitude_deg["pitch"] = degrees(tower.vehicle.attitude.pitch)
    attitude_deg["yaw"] = degrees(tower.vehicle.attitude.yaw)
    return attitude_deg


def get_vehicle_status():
    """Laundry list function.

    """
    if not vehicle_config_data:
        return
    status = {}
    status['battery'] = get_battery()
    status['uptime'] = tower.get_uptime()
    status['armed'] = tower.vehicle.armed
    status['mode'] = tower.vehicle.mode.name
    status['state'] = tower.STATE
    status['altitude'] = tower.vehicle.location.global_relative_frame.alt
    status['attitude'] = get_attitude()
    status['airspeed'] = tower.vehicle.airspeed
    status['velocity'] = get_velocities()
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
            global tower
            if(tower == None):
                tower = Tower()
                tower.initialize()
            elif not tower.vehicle_initialized:
                tower.initialize()
            emit('status', get_vehicle_status())
            return

if __name__ == '__main__':
    socketio.run(app, debug=True)
