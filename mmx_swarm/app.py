from flask import Flask, jsonify, request
from .memory_core import SessionStore


def create_app(db_path: str = "memory_lane.db") -> Flask:
    app = Flask(__name__)
    store = SessionStore(db_path=db_path, keywords=["AI"])

    @app.route("/memory/session")
    def memory_session():
        entries = store.get_recent(100)
        return jsonify(entries)

    @app.route("/memory/tag", methods=["POST"])
    def memory_tag():
        data = request.get_json(force=True)
        entry_id = int(data.get("id"))
        tag = data.get("tag")
        store.add_tag(entry_id, tag)
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
