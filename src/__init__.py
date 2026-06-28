"""
Decomposition UMAP Package
==========================

This package provides a streamlined workflow for applying dimensionality
reduction on complex data by first decomposing it into components and then
using UMAP for embedding.

Exposed Functions:
- decompose_and_embed: A high-level function to train a new model from raw data.
- decompose_with_existing_model: A high-level function to apply a pre-trained model.
- build_interactive_umap: Build a click-to-similarity HTML visualizer from a UMAP XYZ .npy.
- build_interactive_umap_from_embed_map: Convenience wrapper for embed_map lists.
- open_interactive_umap: Same as above but auto-opens in browser.

Exposed Classes:
- DecompositionUMAP: The core class managing the decomposition and UMAP pipeline.
"""
from .decomposition_umap import DecompositionUMAP, decompose_and_embed, decompose_with_existing_model
from .interactive import build_interactive_umap, build_interactive_umap_from_embed_map, open_interactive_umap
from . import example

__all__ = [
    "DecompositionUMAP",
    "decompose_and_embed",
    "decompose_with_existing_model",
    "build_interactive_umap",
    "build_interactive_umap_from_embed_map",
    "open_interactive_umap",
    "example",
]
