# Spatial Utilities

The `src/spatial` module provides monitoring and spatial awareness helpers used during inference.

- **mod.rs** – defines `SpatialAwareness`, configuration settings and event structures used for monitoring risk and accountability.

These utilities consume data from the MorphNet pipeline or external sensors and emit `SpatialEvent` entries when thresholds in `SpatialConfig` are exceeded.
