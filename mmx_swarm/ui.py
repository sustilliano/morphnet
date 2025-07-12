from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from .swarm import SwarmConsciousness

swarm = SwarmConsciousness()

def create_app(static_folder='static'):
    app = Flask(__name__, static_folder=static_folder)
    socketio = SocketIO(app, async_mode='threading')

    @app.route('/')
    def index():
        return send_from_directory(static_folder, 'index.html')

    @socketio.on('connect')
    def connect():
        socketio.emit('identity_update', {'identities': [i.name for i in swarm.identities]})

    return app, socketio


if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, debug=True)
