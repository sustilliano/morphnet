# MMX Format Specification

The MMX format stores multimodal tensors used throughout MorphNet. This
document provides a high-level overview of the format. A full reference is
in progress.

## File Layout

Each MMX file begins with a header describing the contained tensor stack,
followed by compressed data blocks.
