# MorphNet Core

The `src/morphnet` directory houses the main neural and template learning logic.

- **model.rs** – definitions of `MorphNet`, template structures and the `MorphNetBuilder` used to configure a network.
- **classification.rs** – placeholder classification routines returning `ClassificationResult` objects.
- **training.rs** – placeholder training entry point.
- **templates.rs** – factory for constructing common geometric templates like quadrupeds and birds.
- **mod.rs** – module exports to re-export items for crate users.

`MorphNet` instances operate on tensor data loaded from MMX files. Templates created by `TemplateFactory` describe expected body plans which can be refined using the patch quilt module.
