from flask import Flask, jsonify, request
from .conversation_processor import ConversationProcessor
from .swarm import SwarmConsciousness
from .memory_core import SessionStore
from .bias_mixer import set_global_bias, get_global_bias

swarm = SwarmConsciousness()
processor = ConversationProcessor()
store = SessionStore()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route('/conversations', methods=['POST'])
    def conversations():
        file = request.files['file']
        session_id = file.filename.split('.')[0]
        path = f'/tmp/{file.filename}'
        file.save(path)
        messages = processor.parse(path)
        vec = processor.extract_full_profile(messages)
        processor.persist(session_id, messages, vec)
        swarm.add_identity(session_id, vec)
        return jsonify({'status': 'ok', 'id': session_id})

    @app.route('/swarm/state')
    def swarm_state():
        swarm.swarm_update()
        return jsonify({
            'positions': swarm.physics.positions,
            'emergence_history': [],
        })

    @app.route('/query', methods=['POST'])
    def query():
        q = request.get_json(force=True).get('query', '')
        result = swarm.generate_collective_response(q)
        result['bias'] = get_global_bias().tolist()
        return jsonify(result)

    @app.route('/memory/session')
    def memory_session():
        return jsonify(store.get_recent())

    @app.route('/memory/tag', methods=['POST'])
    def memory_tag():
        data = request.get_json(force=True)
        store.add_tag(int(data['id']), data['tag'])
        return jsonify({'status': 'ok'})

    @app.route('/bias', methods=['POST'])
    def bias():
        vec = request.get_json(force=True).get('bias', [0]*99)
        set_global_bias(vec)
        return jsonify({'status': 'ok'})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
