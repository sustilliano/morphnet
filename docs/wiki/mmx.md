# MMX Format

The `src/mmx` directory implements the Multimedia Matrix (MMX) container used throughout MorphNet. Key files include:

- **format.rs** – low level routines for reading and writing MMX headers, compression and directory management.
- **api.rs** – high level interface for creating, reading and appending MMX files.
- **chunks.rs** – data structures for tensors, sequences, text blocks, meshes and embeddings stored in chunks.
- **geometric.rs** – utilities for storing geometric template parameters and body plans.

Data flows into and out of MMX files via the `MMXFile` API. The API uses `write_*` and `read_*` methods to serialize structured data using the helpers in `format.rs` and `chunks.rs`.
