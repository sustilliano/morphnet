# Patch Quilt

`src/patch_quilt` provides mesh refinement utilities that stitch together patch data from different sources.

- **mod.rs** – defines `Patch`, `PatchQuilt` and related structures used to collect and query mesh patches. The quilt can be updated with new sensor `Patch` entries and later searched for nearby patches.

Patches typically originate from sensor data or partial reconstructions. After collecting patches, a higher level routine (not yet implemented) would produce a refined mesh or `MeshRefinement` output.
