import sys

from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit

sys.path.append('../../Flight/')

from AirTrafficControl import Tower

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

tower = Tower()

def get_battery():
    return tower.vehicle.batter.voltage

def get_vehicle_status():
    """Laundry list function.

    """
    get_battery()
    status = {}
    status['battery'] = get_battery()
    return status

@app.route('/')
def index():
    return render_template("index.html")

@socketio.on('connect')
def on_connect():
    send('Initialization in progress...')
    emit('status', {'status': get_vehicle_status()})
    # emit('status', {'status': 'hello world'})




if __name__ == '__main__':
    socketio.run(app, debug=True)
