# MMX Swarm Prototype

This directory contains an early prototype for a Swarm-style intelligence system.

Currently implemented components:

- **ImmediateMemory** – in-memory circular buffer of recent messages.
- **SessionStore** – SQLite-backed log with keyword and sentiment tagging.
- **REST API** – simple Flask server exposing session retrieval and tagging.
- **Demo** – script that logs 1,000 messages and demonstrates keyword search.

Run the demo:

```bash
python -m mmx_swarm.demo
```

Run the API server:

```bash
python -m mmx_swarm.app
```
